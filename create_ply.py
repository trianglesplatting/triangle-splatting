# export triangle splatting scene in Stanford PLY format
# sgreen 8/6/2025
import argparse

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from utils.sh_utils import eval_sh


def save_ply(verts, faces, face_colors, face_opacities, face_sigmas, path):
    print(f"{verts.shape[0]} verts, {faces.shape[0]} faces")

    # blender doesn't import per-face attributes,
    # we we copy face colors etc. to be per-vertex
    vert_colors = np.empty((verts.shape[0], 4), dtype=np.uint8)
    vert_sigmas = np.empty((verts.shape[0], 1), dtype=np.float32)
    for f in range(len(faces)):
        color = face_colors[f]
        v = faces[f]
        vert_colors[v, 0] = color[0]
        vert_colors[v, 1] = color[1]
        vert_colors[v, 2] = color[2]
        vert_colors[v, 3] = face_opacities[f]
        vert_sigmas[v] = face_sigmas[f]

    vert_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("alpha", "u1"),
        ("sigma", "f4"),
    ]
    vert_data = np.empty(verts.shape[0], dtype=vert_dtype)
    vert_attribs = np.concatenate((verts, vert_colors, vert_sigmas), axis=1)
    vert_data[:] = list(map(tuple, vert_attribs))  # make into list of tuples

    face_dtype = [
        ("vertex_index", "i4", (3,)),
        # ("red", "u1"),
        # ("green", "u1"),
        # ("blue", "u1"),
    ]
    face_data = np.empty(faces.shape[0], dtype=face_dtype)

    for i in range(len(faces)):
        face_data[i] = (
            faces[i],
            # colors[i][0],
            # colors[i][1],
            # colors[i][2],
        )

    vert_el = PlyElement.describe(vert_data, "vertex")
    face_el = PlyElement.describe(face_data, "face")
    ply = PlyData([vert_el, face_el], comments=["triangle_splatting"], text=False)
    ply.write(path)

    print(f"saved '{path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a checkpoint to a colored PLY file."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the input checkpoint file (e.g., point_cloud_state_dict.pt)",
        required=True,
    )
    parser.add_argument(
        "--output_name", type=str, help="Name of the output PLY file (e.g., mesh.ply)"
    )
    args = parser.parse_args()

    print(f"Loading '{args.checkpoint_path}'")
    sd = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)

    points = sd["triangles_points"]
    features_dc = sd["features_dc"]
    features_rest = sd["features_rest"]

    features = torch.cat((features_dc, features_rest), dim=1)

    num_coeffs = features.shape[1]
    max_sh_degree = int(np.sqrt(num_coeffs) - 1)

    shs = features.permute(0, 2, 1).contiguous()

    centroids = points.mean(dim=1)
    camera_center = torch.zeros(3)
    dirs = centroids - camera_center
    dirs_norm = dirs / dirs.norm(dim=1, keepdim=True)

    rgb = eval_sh(max_sh_degree, shs, dirs_norm)

    colors_f = torch.clamp(rgb + 0.5, 0.0, 1.0)
    colors = (colors_f * 255).to(torch.uint8)

    opacities = torch.sigmoid(sd["opacity"])
    opacities = torch.clamp(opacities, 0.0, 1.0)
    opacities = (opacities * 255).to(torch.uint8)

    sigmas = 0.01 + torch.exp(sd["sigma"])

    all_verts = points.reshape(-1, 3)

    # remove duplicate vertices
    # unique_verts, inv_idx = torch.unique(all_verts, dim=0, return_inverse=True)
    # faces = inv_idx.reshape(-1, 3)

    faces = torch.arange(0, all_verts.shape[0], dtype=torch.int32).reshape(-1, 3)

    save_ply(
        all_verts.detach().cpu().numpy(),
        faces.detach().cpu().numpy(),
        colors.detach().cpu().numpy(),
        opacities.detach().cpu().numpy(),
        sigmas.detach().cpu().numpy(),
        args.output_name,
    )


if __name__ == "__main__":
    main()
