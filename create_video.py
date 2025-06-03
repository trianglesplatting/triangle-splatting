#
# Copyright (C) 2024, Inria, University of Liege, KAUST and University of Oxford
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  jan.held@uliege.be
#


import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from triangle_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from triangle_renderer import TriangleModel
import numpy as np
from utils.render_utils import generate_path, create_videos
import cv2
from PIL import Image

# --- Helper for progressive zoom trajectory ---
def generate_zoom_trajectory(viewpoint_cameras, n_frames=480, zoom_start=0, zoom_duration=120, zoom_intensity=2.0):
    """
    Generate a camera trajectory with a progressive zoom in and out.
    zoom_start: frame index to start zoom in
    zoom_duration: number of frames for zoom in (zoom out will be symmetric)
    zoom_intensity: factor to multiply the focal length at max zoom
    """
    import copy
    traj = generate_path(viewpoint_cameras, n_frames=n_frames)
    # Get original focal length from the first camera
    cam0 = viewpoint_cameras[0]
    orig_fovx = cam0.FoVx
    orig_fovy = cam0.FoVy
    orig_focalx = cam0.image_width / (2 * np.tan(orig_fovx / 2))
    orig_focaly = cam0.image_height / (2 * np.tan(orig_fovy / 2))
    # Compute new focal for each frame
    for i, cam in enumerate(traj):
        cam = copy.deepcopy(cam)
        # Zoom in
        if zoom_start <= i < zoom_start + zoom_duration:
            t = (i - zoom_start) / max(zoom_duration - 1, 1)
            zoom_factor = 1 + t * (zoom_intensity - 1)
        # Zoom out
        elif zoom_start + zoom_duration <= i < zoom_start + 2 * zoom_duration:
            t = (i - (zoom_start + zoom_duration)) / max(zoom_duration - 1, 1)
            zoom_factor = zoom_intensity - t * (zoom_intensity - 1)
        else:
            zoom_factor = 1.0
        # Update focal length and FoV
        new_focalx = orig_focalx * zoom_factor
        new_focaly = orig_focaly * zoom_factor
        new_fovx = 2 * np.arctan(cam.image_width / (2 * new_focalx))
        new_fovy = 2 * np.arctan(cam.image_height / (2 * new_focaly))
        cam.FoVx = new_fovx
        cam.FoVy = new_fovy
        # Update projection matrix
        from utils.graphics_utils import getProjectionMatrix
        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=new_fovx, fovY=new_fovy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        traj[i] = cam
    return traj

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--save_as", default="output_video", type=str)
    args = get_combined_args(parser)
    print("Creating video for " + args.model_path)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    triangles = TriangleModel(dataset.sh_degree)
    scene = Scene(args=dataset,
                  triangles=triangles,
                  init_opacity=None,
                  init_size=None,
                  nb_points=None,
                  set_sigma=None,
                  load_iteration=args.iteration,
                  shuffle=False)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    traj_dir = os.path.join(args.model_path, 'traj')
    os.makedirs(traj_dir, exist_ok=True)

    render_path = os.path.join(traj_dir, "renders")
    os.makedirs(render_path, exist_ok=True)
    
    n_frames = 240*5
    cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)
    
    with torch.no_grad():
        for idx, view in enumerate(tqdm(cam_traj, desc="Rendering progress")):
            rendering = render(view, triangles, pipe, background)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(traj_dir, "renders", '{0:05d}'.format(idx) + ".png"))

    create_videos(base_dir=traj_dir,
                input_dir=traj_dir, 
                out_name='render_traj', 
                num_frames=n_frames)
