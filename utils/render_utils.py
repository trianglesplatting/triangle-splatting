#
# The original code is under the following copyright:
# Copyright (C) 2022, Google LLC
# Licensed under the Apache License, Version 2.0
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

import numpy as np
import os
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
from utils.graphics_utils import getProjectionMatrix

import torch

def normalize(x: np.ndarray) -> np.ndarray:
	"""Normalization helper function."""
	return x / np.linalg.norm(x)

def pad_poses(p: np.ndarray) -> np.ndarray:
	"""Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
	bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
	return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
	"""Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
	return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Recenter poses around the origin."""
	cam2world = average_pose(poses)
	transform = np.linalg.inv(pad_poses(cam2world))
	poses = transform @ pad_poses(poses)
	return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
	"""New pose using average position, z-axis, and up vector of input poses."""
	position = poses[:, :3, 3].mean(0)
	z_axis = poses[:, :3, 2].mean(0)
	up = poses[:, :3, 1].mean(0)
	cam2world = viewmatrix(z_axis, up, position)
	return cam2world

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
              position: np.ndarray) -> np.ndarray:
	"""Construct lookat view matrix."""
	vec2 = normalize(lookdir)
	vec0 = normalize(np.cross(up, vec2))
	vec1 = normalize(np.cross(vec2, vec0))
	m = np.stack([vec0, vec1, vec2, position], axis=1)
	return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
	"""Calculate nearest point to all focal axes in poses."""
	directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
	m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
	mt_m = np.transpose(m, [0, 2, 1]) @ m
	focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
	return focus_pt

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Transforms poses so principal components lie on XYZ axes.

	Args:
	poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

	Returns:
	A tuple (poses, transform), with the transformed poses and the applied
	camera_to_world transforms.
	"""
	t = poses[:, :3, 3]
	t_mean = t.mean(axis=0)
	t = t - t_mean

	eigval, eigvec = np.linalg.eig(t.T @ t)
	# Sort eigenvectors in order of largest to smallest eigenvalue.
	inds = np.argsort(eigval)[::-1]
	eigvec = eigvec[:, inds]
	rot = eigvec.T
	if np.linalg.det(rot) < 0:
		rot = np.diag(np.array([1, 1, -1])) @ rot

	transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
	poses_recentered = unpad_poses(transform @ pad_poses(poses))
	transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

	# Flip coordinate system if z component of y-axis is negative
	if poses_recentered.mean(axis=0)[2, 1] < 0:
		poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
		transform = np.diag(np.array([1, -1, -1, 1])) @ transform

	return poses_recentered, transform
	# points = np.random.rand(3,100)
	# points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
	# (poses_recentered @ points_h)[0]
	# (transform @ pad_poses(poses) @ points_h)[0,:3]
	# import pdb; pdb.set_trace()

	# # Just make sure it's it in the [-1, 1]^3 cube
	# scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
	# poses_recentered[:, :3, 3] *= scale_factor
	# transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

	# return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                        n_frames: int = 120,
                        const_speed: bool = True,
                        z_variation: float = 0.,
                        z_phase: float = 0.) -> np.ndarray:
	"""Generate an elliptical render path based on the given poses."""
	# Calculate the focal point for the path (cameras point toward this).
	center = focus_point_fn(poses)
	# Path height sits at z=0 (in middle of zero-mean capture pattern).
	offset = np.array([center[0], center[1], 0])

	# Calculate scaling for ellipse axes based on input camera positions.
	sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
	# Use ellipse that is symmetric about the focal point in xy.
	low = -sc + offset
	high = sc + offset
	# Optional height variation need not be symmetric
	z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
	z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

	def get_positions(theta):
		# Interpolate between bounds with trig functions to get ellipse in x-y.
		# Optionally also interpolate in z to change camera height along path.
		return np.stack([
			low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
			low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
			z_variation * (z_low[2] + (z_high - z_low)[2] *
							(np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
		], -1)

	theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
	positions = get_positions(theta)

	#if const_speed:

	# # Resample theta angles so that the velocity is closer to constant.
	# lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
	# theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
	# positions = get_positions(theta)

	# Throw away duplicated last position.
	positions = positions[:-1]

	# Set path's up vector to axis closest to average of input pose up vectors.
	avg_up = poses[:, :3, 1].mean(0)
	avg_up = avg_up / np.linalg.norm(avg_up)
	ind_up = np.argmax(np.abs(avg_up))
	up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

	return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_path(viewpoint_cameras, n_frames=480):
	c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
	pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
	pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

	# generate new poses
	new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
	# warp back to orignal scale
	new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

	traj = []
	for c2w in new_poses:
		c2w = c2w @ np.diag([1, -1, -1, 1])
		cam = copy.deepcopy(viewpoint_cameras[0])
		cam.image_height = int(cam.image_height / 2) * 2
		cam.image_width = int(cam.image_width / 2) * 2
		cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()
		cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
		cam.camera_center = cam.world_view_transform.inverse()[3, :3]
		traj.append(cam)

	return traj

def generate_zoom_trajectory(viewpoint_cameras, n_frames=480, zoom_start=0, zoom_duration=120, zoom_intensity=2.0):
    traj = generate_path(viewpoint_cameras, n_frames=n_frames)

    cam0 = viewpoint_cameras[0]
    orig_fovx = cam0.FoVx
    orig_fovy = cam0.FoVy
    orig_focalx = cam0.image_width / (2 * np.tan(orig_fovx / 2))
    orig_focaly = cam0.image_height / (2 * np.tan(orig_fovy / 2))

    for i, cam in enumerate(traj):
        cam = copy.deepcopy(cam)

        if zoom_start <= i < zoom_start + zoom_duration:
            t = (i - zoom_start) / max(zoom_duration - 1, 1)
            zoom_factor = 1 + t * (zoom_intensity - 1)

        elif zoom_start + zoom_duration <= i < zoom_start + 2 * zoom_duration:
            t = (i - (zoom_start + zoom_duration)) / max(zoom_duration - 1, 1)
            zoom_factor = zoom_intensity - t * (zoom_intensity - 1)
        else:
            zoom_factor = 1.0

        new_focalx = orig_focalx * zoom_factor
        new_focaly = orig_focaly * zoom_factor
        new_fovx = 2 * np.arctan(cam.image_width / (2 * new_focalx))
        new_fovy = 2 * np.arctan(cam.image_height / (2 * new_focaly))
        cam.FoVx = new_fovx
        cam.FoVy = new_fovy

        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=new_fovx, fovY=new_fovy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        traj[i] = cam
    return traj

def load_img(pth: str) -> np.ndarray:
	"""Load an image and cast to float32."""
	with open(pth, 'rb') as f:
		image = np.array(Image.open(f), dtype=np.float32)
	return image


def create_videos(base_dir, input_dir, out_name, num_frames=480):
	"""Creates videos out of the images saved to disk."""
	# Last two parts of checkpoint path are experiment name and scene name.
	video_prefix = f'{out_name}'

	zpad = max(5, len(str(num_frames - 1)))
	idx_to_str = lambda idx: str(idx).zfill(zpad)

	os.makedirs(base_dir, exist_ok=True)

	img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.png')
	img = load_img(img_file)
	shape = img.shape

	print(f'Video shape is {shape[:2]}')

	video_kwargs = {
		'shape': shape[:2],
		'codec': 'h264',
		'fps': 120,
		'crf': 18,
	}

	video_file = os.path.join(base_dir, f'{video_prefix}_color.mp4')
	input_format = 'rgb'

	file_ext = 'png'
	idx = 0

	file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')

	if not os.path.exists(file0):
		return
	print(f'Making video {video_file}...')
	with media.VideoWriter(video_file, **video_kwargs, input_format=input_format) as writer:
		for idx in tqdm(range(num_frames)):

			img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')

			if not os.path.exists(img_file):
				ValueError(f'Image file {img_file} does not exist.')
			img = load_img(img_file)
			img = img / 255.

			frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
			writer.add_image(frame)
			idx += 1

def save_img_u8(img, pth):
	"""Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
	with open(pth, 'wb') as f:
		Image.fromarray(
			(np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
				f, 'PNG')
		
def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')
