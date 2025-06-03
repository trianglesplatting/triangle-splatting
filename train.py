#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, equilateral_regularizer, l2_loss
from triangle_renderer import render
import sys
from scene import Scene, TriangleModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import lpips


def training(
        dataset,   
        opt, 
        pipe,
        no_dome, 
        outdoor,
        testing_iterations,
        save_iterations,
        checkpoint, 
        debug_from,
        ):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Load parameters, triangles and scene
    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(dataset, triangles, opt.set_opacity, opt.triangle_size, opt.nb_points, opt.set_sigma, no_dome)
    triangles.training_setup(opt, opt.lr_mask, opt.feature_lr, opt.opacity_lr, opt.lr_sigma, opt.lr_triangles_points_init)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        triangles.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    number_of_views = len(viewpoint_stack)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    total_dead = 0

    opacity_now = True

    new_round = False
    removed_them = False

    large_scene = triangles.large

    if large_scene and outdoor:
        loss_fn = l2_loss
    else:
        loss_fn = l1_loss

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        triangles.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            triangles.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            if not new_round and removed_them:
                new_round = True
                removed_them = False
            else:
                new_round = False

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, triangles, pipe, bg)
        image = render_pkg["render"]

        # largest distance from point to center of image
        triangle_area = render_pkg["density_factor"].detach()
        # largest distance from point after applying sigma to center of image
        image_size = render_pkg["scaling"].detach()
        importance_score = render_pkg["max_blending"].detach()

        if new_round:
            mask = triangle_area > 1
            triangles.triangle_area[mask] += 1

        mask = image_size > triangles.image_size
        triangles.image_size[mask] = image_size[mask]
        mask = importance_score > triangles.importance_score
        triangles.importance_score[mask] = importance_score[mask]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        pixel_loss = loss_fn(image, gt_image)

        ##############################################################
        # WE ADD A LOSS FORCING LOW OPACITIES                        #
        ##############################################################
        loss_image = (1.0 - opt.lambda_dssim) * pixel_loss + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # loss opacity
        loss_opacity = torch.abs(triangles.get_opacity).mean() * args.lambda_opacity

        # loss normal and distortion
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        lambda_dist = opt.lambda_dist if iteration > opt.iteration_mesh else 0
        lambda_normal = opt.lambda_normals if iteration > opt.iteration_mesh else 0 # 0.001
        rend_dist = render_pkg["rend_dist"]
        dist_loss = lambda_dist * (rend_dist).mean()
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()

        loss_size = 1 / equilateral_regularizer(triangles.get_triangles_points).mean() 
        loss_size = loss_size * opt.lambda_size


        if iteration < opt.densify_until_iter:
            loss = loss_image + loss_opacity + normal_loss + dist_loss + loss_size
        else:
            loss = loss_image + loss_opacity + normal_loss + dist_loss

        loss.backward()
     
        iter_end.record()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            
            training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration in save_iterations:
                print("\n[ITER {}] Saving Triangles".format(iteration))
                scene.save(iteration)
            if iteration % 1000 == 0:
                total_dead = 0

            if iteration < opt.densify_until_iter and iteration % opt.densification_interval == 0 and iteration > opt.densify_from_iter:
                
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= args.opacity_dead).squeeze()

                if iteration > 1000 and not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                    
                    if not outdoor:
                        mask_test = triangles.image_size > 1400
                        dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                          

                total_dead += dead_mask.sum()

                if opt.proba_distr == 0:
                    oddGroup = True
                elif opt.proba_distr == 1:
                    oddGroup = False
                else:
                    if opacity_now:
                        oddGroup = opacity_now
                        opacity_now = False
                    else:
                        oddGroup = opacity_now
                        opacity_now = True

                removed_them = True
                new_round = False

                triangles.add_new_gs(cap_max=opt.max_shapes, oddGroup=oddGroup, dead_mask=dead_mask)


            if iteration > opt.densify_until_iter and iteration % opt.densification_interval == 0:
                if number_of_views < 250:
                    dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                else:
                    if not new_round:
                        dead_mask = torch.logical_or((triangles.importance_score < args.importance_threshold).squeeze(),(triangles.get_opacity <= args.opacity_dead).squeeze())
                    else:
                        dead_mask = (triangles.get_opacity <= args.opacity_dead).squeeze()


                if not new_round:
                    mask_test = triangles.triangle_area < 2
                    dead_mask = torch.logical_or(dead_mask, mask_test.squeeze())
                triangles.remove_final_points(dead_mask)
                removed_them = True
                new_round = False

            if iteration < opt.iterations:
                triangles.optimizer.step()
                triangles.optimizer.zero_grad(set_to_none = True)
                
    print("Training is done")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, pixel_loss, loss, loss_fn, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/pixel_loss', pixel_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                pixel_loss_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                total_time = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    image = torch.clamp(renderFunc(viewpoint, scene.triangles, *renderArgs)["render"], 0.0, 1.0)
                    end_event.record()
                    torch.cuda.synchronize()
                    runtime = start_event.elapsed_time(end_event)
                    total_time += runtime

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    pixel_loss_test += loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                pixel_loss_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                total_time /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], pixel_loss_test, psnr_test, ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', pixel_loss_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.triangles.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.triangles.get_triangles_points.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--no_dome", action="store_true", default=False)
    parser.add_argument("--outdoor", action="store_true", default=False)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args.no_dome,
             args.outdoor,
             args.test_iterations,
             args.save_iterations,
             args.start_checkpoint,
             args.debug_from,
             )
    
    # All done
    print("\nTraining complete.")