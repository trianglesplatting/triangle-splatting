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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
import math


def random_rotation_matrices(num_matrices, device='cpu'):
    """
    Returns a tensor of shape (num_matrices, 3, 3) containing 
    random 3D rotation matrices.
    """
    axis = torch.randn(num_matrices, 3, device=device)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    
    angles = 2.0 * math.pi * torch.rand(num_matrices, device=device)
    sin_t = torch.sin(angles)
    cos_t = torch.cos(angles)
    
    K = torch.zeros(num_matrices, 3, 3, device=device)
    ux, uy, uz = axis[:, 0], axis[:, 1], axis[:, 2]
    K[:, 0, 1] = -uz
    K[:, 0, 2] =  uy
    K[:, 1, 0] =  uz
    K[:, 1, 2] = -ux
    K[:, 2, 0] = -uy
    K[:, 2, 1] =  ux
    
    K2 = K.bmm(K)
    
    I = torch.eye(3, device=device).unsqueeze(0).expand(num_matrices, -1, -1)
    
    sin_term = sin_t.view(-1, 1, 1) * K
    cos_term = (1.0 - cos_t).view(-1, 1, 1) * K2
    
    return I + sin_term + cos_term


def fibonacci_directions(nb_points, device='cpu'):
    """
    Generate nb_points points on the unit sphere using a Fibonacci approach.
    Returns a tensor of shape (nb_points, 3).
    """
    directions = []
    for i in range(nb_points):
        z_coord = 1.0 - (2.0 * i / (nb_points - 1))
        z_coord = torch.tensor(z_coord, device=device)
        radius_xy = torch.sqrt(1.0 - z_coord * z_coord)
        theta = math.pi * (3.0 - math.sqrt(5.0)) * i
        
        x_unit = radius_xy * torch.cos(torch.tensor(theta, device=device))
        y_unit = radius_xy * torch.sin(torch.tensor(theta, device=device))
        
        directions.append(torch.stack([x_unit, y_unit, z_coord]))
    return torch.stack(directions, dim=0)


def generate_triangles_in_chunks(x, y, z, radii, nb_points=3, chunk_size=2000):
    device = x.device

    num_centers = x.shape[0]

    base_dirs = fibonacci_directions(nb_points, device=device)
    out_points = torch.zeros(num_centers, nb_points, 3, device=device)

    for start_idx in range(0, num_centers, chunk_size):
        end_idx = min(start_idx + chunk_size, num_centers)

        x_chunk = x[start_idx:end_idx]
        y_chunk = y[start_idx:end_idx]
        z_chunk = z[start_idx:end_idx]
        r_chunk = radii[start_idx:end_idx]

        chunk_size_actual = x_chunk.shape[0]

        R_chunk = random_rotation_matrices(chunk_size_actual, device=device)

        for i in range(nb_points):
            dir_i = base_dirs[i]

            dir_i_expanded = dir_i.view(1, 3, 1).expand(chunk_size_actual, -1, -1)

            rotated = R_chunk.bmm(dir_i_expanded)
            rotated = rotated.squeeze(-1)

            scaled = rotated * r_chunk.view(-1, 1)

            centers = torch.stack([x_chunk, y_chunk, z_chunk], dim=1)

            result_pts = centers + scaled

            out_points[start_idx:end_idx, i, :] = result_pts

    return out_points


def densify_pcd_on_box(pcd: BasicPointCloud, num_new_points: int, scale_box: float = 1.0):
    """
    Densify the point cloud by sampling new points on all 6 faces of its (optionally scaled) bounding box.
    No assumption is made about axis orientation.
    Colors are assigned by nearest neighbor from the original point cloud.
    """
    import torch
    import numpy as np
    from scipy.spatial import cKDTree

    points_np = np.asarray(pcd.points)
    colors_np = np.asarray(pcd.colors)
    device = "cuda"

    min_xyz = points_np.min(axis=0)
    max_xyz = points_np.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0
    half_size = (max_xyz - min_xyz) / 2.0
    half_size_scaled = half_size * scale_box
    min_xyz_scaled = center - half_size_scaled
    max_xyz_scaled = center + half_size_scaled

    faces = []
    axes = [0, 1, 2]
    for axis in axes:
        for is_min in [True, False]:
            fixed_val = min_xyz_scaled[axis] if is_min else max_xyz_scaled[axis]

            free_axes = [a for a in axes if a != axis]
            faces.append((axis, fixed_val, free_axes[0], free_axes[1]))

    num_faces = len(faces)
    points_per_face = num_new_points // num_faces
    extras = num_new_points % num_faces

    new_points = []
    for i, (fixed_axis, fixed_val, var_axis1, var_axis2) in enumerate(faces):
        n = points_per_face + (1 if i < extras else 0)
        v1_min, v1_max = min_xyz_scaled[var_axis1], max_xyz_scaled[var_axis1]
        v2_min, v2_max = min_xyz_scaled[var_axis2], max_xyz_scaled[var_axis2]
        v1 = np.random.uniform(v1_min, v1_max, n)
        v2 = np.random.uniform(v2_min, v2_max, n)
        pts = np.zeros((n, 3))
        pts[:, fixed_axis] = fixed_val
        pts[:, var_axis1] = v1
        pts[:, var_axis2] = v2
        new_points.append(pts)
    new_points = np.concatenate(new_points, axis=0)

    kdtree = cKDTree(points_np)
    _, idxs = kdtree.query(new_points, k=1)
    new_colors = colors_np[idxs]

    new_points_torch = torch.tensor(new_points, dtype=torch.float32, device=device)
    new_colors_torch = torch.tensor(new_colors, dtype=torch.float32, device=device)

    return new_points_torch, new_colors_torch


class TriangleModel:

    def setup_functions(self):
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.exponential_activation = lambda x: 0.01 + torch.exp(x)
        self.inverse_exponential_activation = lambda y: torch.log(y - 0.01)

    def __init__(self, sh_degree : int):
        self._triangles_points = torch.empty(0)
        self._sigma = torch.empty(0)
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_density_factor = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

        self._mask = torch.empty(0)

        self.max_scaling = torch.empty(0)

        self._num_points_per_triangle = torch.empty(0)
        self._cumsum_of_points_per_triangle = torch.empty(0)
        self._number_of_points = 0

        self.split_size = 0

        self.triangle_area = 0
        self.image_size = 0
        self.importance_score = 0

        self.nb_points = 0

        self.large = False

        self.setup_functions()

    def save(self, path):

        mkdir_p(path)

        point_cloud_state_dict = {}

        point_cloud_state_dict["triangles_points"] = self._triangles_points
        point_cloud_state_dict["sigma"] = self._sigma
        point_cloud_state_dict["active_sh_degree"] = self.active_sh_degree
        point_cloud_state_dict["features_dc"] = self._features_dc
        point_cloud_state_dict["features_rest"] = self._features_rest
        point_cloud_state_dict["opacity"] = self._opacity

        torch.save(point_cloud_state_dict, os.path.join(path, 'point_cloud_state_dict.pt'))

        hyperparameters = {}

        hyperparameters["max_radii2D"] = self.max_radii2D
        hyperparameters["denom"] = self.denom
        hyperparameters["spatial_lr_scale"] = self.spatial_lr_scale
        hyperparameters["num_points_per_triangle"] = self._num_points_per_triangle
        hyperparameters["cumsum_of_points_per_triangle"] = self._cumsum_of_points_per_triangle
        hyperparameters["number_of_points"] = self._number_of_points
        hyperparameters["max_scaling"] = self.max_scaling
        hyperparameters["max_density_factor"] = self.max_density_factor

        torch.save(hyperparameters, os.path.join(path, 'hyperparameters.pt'))

    def load(self, path):

        point_cloud_state_dict = torch.load(os.path.join(path, 'point_cloud_state_dict.pt'))

        shapes = point_cloud_state_dict["triangles_points"]
        max_shape = shapes.shape[0]
        print(f"Loaded {max_shape} triangle shapes")

        i = 0
        plus = max_shape

        self._triangles_points = point_cloud_state_dict["triangles_points"][i:i+plus].to("cuda").to(torch.float32).detach().clone().requires_grad_(True)
        self._sigma = point_cloud_state_dict["sigma"][i:i+plus].to("cuda").to(torch.float32).detach().clone().requires_grad_(True)
        self.active_sh_degree = point_cloud_state_dict["active_sh_degree"] 
        self._features_dc = point_cloud_state_dict["features_dc"][i:i+plus].to("cuda").to(torch.float32).detach().clone().requires_grad_(True)
        self._features_rest = point_cloud_state_dict["features_rest"][i:i+plus].to("cuda").to(torch.float32).detach().clone().requires_grad_(True)
        self._opacity = point_cloud_state_dict["opacity"][i:i+plus].to("cuda").to(torch.float32).detach().clone().requires_grad_(True)
        
        self._mask = nn.Parameter(torch.ones((self._triangles_points.size(0), 1), device="cuda").requires_grad_(True))
        num_points_per_triangle = []
        for i in range(self._triangles_points.size(0)):
            num_points_per_triangle.append(self._triangles_points[i].shape[0])
        tensor_num_points_per_triangle = torch.tensor(num_points_per_triangle, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_triangle = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_triangle, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = self._triangles_points.shape[0]


        self._num_points_per_triangle = tensor_num_points_per_triangle
        self._cumsum_of_points_per_triangle = cumsum_of_points_per_triangle
        self._number_of_points = number_of_points

        l = [
            {'params': [self._features_dc], 'lr': 0.00001, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.00001 / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': 0.00001, "name": "opacity"},
            {'params': [self._triangles_points], 'lr': 0.00001, "name": "triangles_points"},
            {'params': [self._sigma], 'lr': 0.00001, "name": "sigma"},
            {'params': [self._mask], 'lr':  0.00001, "name": "mask"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    @property
    def get_triangles_points_flatten(self):
        return self._triangles_points.flatten(0)
  
    @property
    def get_triangles_points(self):
        return self._triangles_points
    
    @property
    def get_max_scaling(self):
        return self.max_scaling
    
    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_num_points_per_triangle(self):
        return self._num_points_per_triangle
    
    @property
    def get_cumsum_of_points_per_triangle(self):
        return self._cumsum_of_points_per_triangle
    
    @property
    def get_number_of_points(self):
        return self._number_of_points

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def fibonacci_sphere(self, samples=1000):
        points = []
        phi = math.pi * (math.sqrt(5.) - 1.)

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2
            radius = math.sqrt(1 - y * y)

            theta = phi * i 

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append((x, y, z))

        return np.asarray(points)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, opacity : float, init_size : float, nb_points: int, set_sigma : float, no_dome: bool):

        self.spatial_lr_scale = spatial_lr_scale
        pcd_points = np.asarray(pcd.points)
        total_number_of_points = pcd_points.shape[0]
        shapes_to_add = int(total_number_of_points * 0.05)
        radius = np.max(np.abs(pcd_points))
        sky_box_points = self.fibonacci_sphere(shapes_to_add) * radius
        pcd_colors = np.asarray(pcd.colors)

        if no_dome:
            total_points = pcd_points
            total_colors = pcd_colors
        else:
            total_points = np.concatenate([sky_box_points, pcd_points], axis=0)
            total_colors = np.concatenate([np.ones_like(sky_box_points), pcd_colors], axis=0)

        self.nb_points = nb_points
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(total_points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(total_colors)).float().cuda())

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        x, y, z = fused_point_cloud[:, 0], fused_point_cloud[:, 1], fused_point_cloud[:, 2]

        x_min, x_max = torch.min(x), torch.max(x)
        y_min, y_max = torch.min(y), torch.max(y)
        z_min, z_max = torch.min(z), torch.max(z)
        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min
        scene_size = max(width, height, depth)

        if scene_size > 300:
            print("Scene is large, we increase the threshold")
            self.large = True

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(total_points)).float().cuda()), 0.0000001)
        
        radii = init_size * torch.sqrt(dist2).unsqueeze(1)

        points_per_triangle = generate_triangles_in_chunks(x, y, z, radii, nb_points)

        num_points_per_triangle = []
        for i in range(points_per_triangle.size(0)):
            num_points_per_triangle.append(points_per_triangle[i].shape[0])
        tensor_num_points_per_triangle = torch.tensor(num_points_per_triangle, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_triangle = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_triangle, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = points_per_triangle.shape[0]

        opacities = inverse_sigmoid(opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        sigmas = self.inverse_exponential_activation(torch.ones((number_of_points, 1), dtype=torch.float, device="cuda") *  set_sigma)
                    
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._triangles_points = nn.Parameter(points_per_triangle.to('cuda').requires_grad_(True))
        self._sigma = nn.Parameter(sigmas.requires_grad_(True))
        self._num_points_per_triangle = tensor_num_points_per_triangle
        self._cumsum_of_points_per_triangle = cumsum_of_points_per_triangle
        self._number_of_points = number_of_points
        self.max_scaling = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")
        self.max_radii2D = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")
        self.max_density_factor = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self.triangle_area = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")
        self.image_size = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")
        self.importance_score = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")

    def training_setup(self, training_args, lr_mask, lr_features, lr_opacity, lr_sigma, lr_triangles_points_init):

        self.denom = torch.zeros((self.get_triangles_points.shape[0], 1), device="cuda")

        self.split_size = training_args.split_size
        self.lr_sigma = lr_sigma
        self.start_lr_sigma = training_args.start_lr_sigma
        self.max_noise_factor = training_args.max_noise_factor

        self.add_shape = training_args.add_shape

        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': lr_opacity, "name": "opacity"},
            {'params': [self._triangles_points], 'lr': lr_triangles_points_init, "name": "triangles_points"},
            {'params': [self._sigma], 'lr': lr_sigma, "name": "sigma"},
            {'params': [self._mask], 'lr':  0, "name": "mask"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.triangle_scheduler_args = get_expon_lr_func(lr_init=lr_triangles_points_init,
                                                        lr_final=lr_triangles_points_init/100,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)


    def reset_sigma(self):
        new_sigma = self.inverse_exponential_activation(torch.ones((self._opacity.shape[0], 1), dtype=torch.float, device="cuda"))
        optimizable_tensors = self.replace_tensor_to_optimizer(new_sigma, "sigma")
        self._sigma = optimizable_tensors["sigma"]
        print("changed sigma to 1")

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "triangles_points":
                    lr = self.triangle_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    

        return optimizable_tensors
    
    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {"triangles_points": self._triangles_points,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "sigma" : self._sigma,
            "mask": self._mask}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._triangles_points = optimizable_tensors["triangles_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._sigma = optimizable_tensors["sigma"] 
        self._mask = optimizable_tensors["mask"]

        torch.cuda.empty_cache()
        
        return optimizable_tensors

    def densification_postfix(self, new_triangles_points, new_features_dc, new_features_rest, new_opacities, new_sigma, new_mask):
        d = {"triangles_points": new_triangles_points,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "sigma" : new_sigma,
        "mask": new_mask}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._triangles_points = optimizable_tensors["triangles_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._sigma = optimizable_tensors["sigma"]
        self._mask = optimizable_tensors["mask"]

        self.denom = torch.zeros((self.get_triangles_points.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")
        self.max_density_factor = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")
        self.triangle_area = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")

        self.max_scaling = torch.cat((self.max_scaling, torch.zeros(new_opacities.shape[0], device="cuda")),dim=0)

        num_points_per_triangle = []
        for i in range(self._triangles_points.size(0)):
            num_points_per_triangle.append(self._triangles_points[i].shape[0])
        tensor_num_points_per_triangle = torch.tensor(num_points_per_triangle, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_triangle = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_triangle, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = self._triangles_points.shape[0]

        self._num_points_per_triangle = tensor_num_points_per_triangle
        self._cumsum_of_points_per_triangle = cumsum_of_points_per_triangle
        self._number_of_points = number_of_points

    def _update_params_small(self, idxs):
        new_triangles_points = self._triangles_points[idxs]
        n = new_triangles_points.shape[0]

        v1 = new_triangles_points[:, 1] - new_triangles_points[:, 0]
        v2 = new_triangles_points[:, 2] - new_triangles_points[:, 0]
        normals = torch.cross(v1, v2, dim=1)                  
        normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-9) 

        min_coords = new_triangles_points.min(dim=1).values
        max_coords = new_triangles_points.max(dim=1).values
        shape_sizes = max_coords - min_coords

        max_noise_factor = self.max_noise_factor
        noise_scale = shape_sizes * max_noise_factor
        noise = (torch.rand(n, 1, 3, device=new_triangles_points.device) - 0.5) * noise_scale.unsqueeze(1)

        dot_products = (noise * normals.unsqueeze(1)).sum(dim=-1, keepdim=True)
        noise_in_plane = noise - dot_products * normals.unsqueeze(1)

        new_triangles_points_noisy = new_triangles_points + noise_in_plane

        opacity_old = self.get_opacity[idxs]
        opacity_new = inverse_sigmoid(1.0 - torch.pow(1.0 - opacity_old, 1.0 / 2))
        
        return (torch.cat([self._triangles_points[idxs], new_triangles_points_noisy], dim=0),
                torch.cat([self._features_dc[idxs], self._features_dc[idxs]], dim=0),
                torch.cat([self._features_rest[idxs], self._features_rest[idxs]], dim=0),
                torch.cat([opacity_new, opacity_new], dim=0),
                torch.cat([self._sigma[idxs], self._sigma[idxs]], dim=0),
                torch.cat([self._mask[idxs], torch.ones((n, 1), device=self._mask.device)], dim=0))

    def _update_params(self, selected_indices):
        selected_triangles_points = self._triangles_points[selected_indices]

        A = selected_triangles_points[:, 0, :]
        B = selected_triangles_points[:, 1, :]
        C = selected_triangles_points[:, 2, :]

        M_AB = (A + B) / 2
        M_AC = (A + C) / 2
        M_BC = (B + C) / 2

        sub1 = torch.stack([A, M_AB, M_AC], dim=1)
        sub2 = torch.stack([B, M_AB, M_BC], dim=1)
        sub3 = torch.stack([C, M_AC, M_BC], dim=1)
        sub4 = torch.stack([M_AB, M_AC, M_BC], dim=1)

        new_triangles_points = torch.cat([sub1, sub2, sub3, sub4], dim=0)

        new_features_dc = self._features_dc[selected_indices].repeat(4, 1, 1)
        new_features_rest = self._features_rest[selected_indices].repeat(4, 1, 1)
        new_opacities = self._opacity[selected_indices].repeat(4, 1)
        new_sigma = self._sigma[selected_indices].repeat(4, 1)
        new_mask = torch.ones_like(self._mask[selected_indices].repeat(4, 1))

        return new_triangles_points, new_features_dc, new_features_rest, new_opacities, new_sigma, new_mask

    def _sample_alives(self, probs, num, big_mask, alive_indices=None):
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = torch.clamp(probs, min=0.0)
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, min(num, (probs>0).sum().item()), replacement=False)

        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]

        device = probs.device
        cost_true = torch.tensor(3, dtype=torch.int64, device=device)
        cost_false = torch.tensor(1, dtype=torch.int64, device=device)
        costs = torch.where(big_mask[sampled_idxs], cost_true, cost_false)
        
        cum_costs = torch.cumsum(costs, dim=0)
        
        cutoff_idx = (cum_costs >= num).nonzero(as_tuple=True)[0]
        if cutoff_idx.numel() > 0:
            cutoff = cutoff_idx[0].item() + 1 
        else:
            cutoff = sampled_idxs.numel()
        return sampled_idxs[:cutoff]

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._triangles_points = optimizable_tensors["triangles_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._sigma = optimizable_tensors["sigma"]
        self._mask = optimizable_tensors["mask"]

        num_points_per_triangle = []
        for i in range(self._triangles_points.size(0)):
            num_points_per_triangle.append(self._triangles_points[i].shape[0])
        tensor_num_points_per_triangle = torch.tensor(num_points_per_triangle, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_triangle = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_triangle, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = self._triangles_points.shape[0]

        self._num_points_per_triangle = tensor_num_points_per_triangle
        self._cumsum_of_points_per_triangle = cumsum_of_points_per_triangle
        self._number_of_points = number_of_points

    def add_new_gs(self, cap_max, oddGroup=True, dead_mask=None):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(self.add_shape * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        num_gs += dead_mask.sum()

        if num_gs <= 0:
            return 0

        if oddGroup:
            probs = self.get_opacity.squeeze(-1) 
        else:
            eps = torch.finfo(torch.float32).eps
            probs = self.get_sigma.squeeze(-1) 
            probs = 1 / (probs + eps)
            
        probs[dead_mask] = 0

        compar = self.image_size
        big_mask   = compar > self.split_size

        add_idx = self._sample_alives(probs=probs, num=num_gs, big_mask=big_mask)

        big_mask   = compar[add_idx] > self.split_size
        small_mask = ~big_mask
        big_indices   = add_idx[big_mask]
        small_indices = add_idx[small_mask]

        num_big = big_indices.shape[0]
        if num_big > 0:

            (split_triangles_points,
            split_features_dc,
            split_features_rest,
            split_opacity,
            split_sigma,
            split_mask) = self._update_params(big_indices)

        else:
            split_triangles_points  = torch.empty((0, 3, 3),   device=self._triangles_points.device)
            split_features_dc    = torch.empty((0,) + self._features_dc.shape[1:],   device=self._features_dc.device)
            split_features_rest  = torch.empty((0,) + self._features_rest.shape[1:], device=self._features_rest.device)
            split_opacity        = torch.empty((0, 1), device=self._opacity.device)
            split_sigma          = torch.empty((0, 1), device=self._sigma.device)
            split_mask           = torch.empty((0, 1), device=self._mask.device)


        num_small = small_indices.shape[0]
        if num_small > 0:
            (clone_triangles_points,
            clone_features_dc,
            clone_features_rest,
            clone_opacity,
            clone_sigma,
            clone_mask) = self._update_params_small(small_indices)

        else:
            clone_triangles_points  = torch.empty((0, 3, 3),   device=self._triangles_points.device)
            clone_features_dc    = torch.empty((0,) + self._features_dc.shape[1:],   device=self._features_dc.device)
            clone_features_rest  = torch.empty((0,) + self._features_rest.shape[1:], device=self._features_rest.device)
            clone_opacity        = torch.empty((0, 1), device=self._opacity.device)
            clone_sigma          = torch.empty((0, 1), device=self._sigma.device)
            clone_mask           = torch.empty((0, 1), device=self._mask.device)

        new_triangles_points = torch.cat([split_triangles_points, clone_triangles_points], dim=0)
        new_features_dc   = torch.cat([split_features_dc,   clone_features_dc],   dim=0)
        new_features_rest = torch.cat([split_features_rest, clone_features_rest], dim=0)
        new_opacity       = torch.cat([split_opacity,       clone_opacity],       dim=0)
        new_sigma         = torch.cat([split_sigma,         clone_sigma],         dim=0)
        new_mask          = torch.cat([split_mask,          clone_mask],          dim=0)

        self.densification_postfix(new_triangles_points, new_features_dc, new_features_rest, new_opacity, new_sigma, new_mask)
        self.replace_tensors_to_optimizer(inds=add_idx)

        mask = torch.zeros(self._opacity.shape[0], dtype=torch.bool)
        mask[add_idx] = True
        mask[torch.nonzero(dead_mask, as_tuple=True)] = True
        self.prune_points(mask)

        self.triangle_area = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")
        self.image_size = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")
        self.importance_score = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")

    def remove_final_points(self, mask):
        self.prune_points(mask)
        self.triangle_area = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")
        self.image_size = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")
        self.importance_score = torch.zeros((self.get_triangles_points.shape[0]), device="cuda")


    def reset_opacity(self, sigma_reset):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*sigma_reset))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def get_attributes_by_indices(self, indices):

        return {
            "triangles_points": self._triangles_points[indices].detach().clone(),
            "sigma": self._sigma[indices].detach().clone(),
            "features_dc": self._features_dc[indices].detach().clone(),
            "features_rest": self._features_rest[indices].detach().clone(),
            "opacity": self._opacity[indices].detach().clone(),
            "triangle_area": self.triangle_area[indices].detach().clone() if hasattr(self, 'triangle_area') else None,
            "mask": self._mask[indices].detach().clone() if hasattr(self, '_mask') else None
        }

    def get_model_by_indices(self, indices):

        new_model = TriangleModel(self.max_sh_degree)

        new_model._triangles_points = self._triangles_points[indices].detach().clone()
        new_model._sigma = self._sigma[indices].detach().clone()
        new_model._features_dc = self._features_dc[indices].detach().clone()
        new_model._features_rest = self._features_rest[indices].detach().clone()
        new_model._opacity = self._opacity[indices].detach().clone()
        new_model.active_sh_degree = self.active_sh_degree

        if hasattr(self, '_mask') and self._mask is not None:
            new_model._mask = self._mask[indices].detach().clone()

        num_points_per_triangle = []
        for i in range(new_model._triangles_points.size(0)):
            num_points_per_triangle.append(new_model._triangles_points[i].shape[0])

        tensor_num_points_per_triangle = torch.tensor(num_points_per_triangle, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_triangle = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_triangle, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = new_model._triangles_points.shape[0]

        new_model._num_points_per_triangle = tensor_num_points_per_triangle
        new_model._cumsum_of_points_per_triangle = cumsum_of_points_per_triangle
        new_model._number_of_points = number_of_points

        new_model.max_scaling = torch.zeros((number_of_points), dtype=torch.float, device="cuda")
        new_model.max_radii2D = torch.zeros((number_of_points), dtype=torch.float, device="cuda")
        new_model.max_density_factor = torch.zeros((number_of_points), dtype=torch.float, device="cuda")
        new_model.denom = torch.zeros((number_of_points, 1), device="cuda")

        return new_model
