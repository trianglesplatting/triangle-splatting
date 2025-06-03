import torch
import numpy as np
from utils.sh_utils import eval_sh
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert a checkpoint to a colored OFF file.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the input checkpoint file (e.g., point_cloud_state_dict.pt)")
    parser.add_argument("--output_name", type=str, help="Name of the output OFF file (e.g., mesh_colored.off)")
    args = parser.parse_args()

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

    all_verts = points.reshape(-1, 3)
    unique_verts, inv_idx = torch.unique(all_verts, dim=0, return_inverse=True)
    faces = inv_idx.reshape(-1, 3)

    with open(args.output_name, "w") as f:
        f.write("COFF\n")
        f.write(f"{len(unique_verts)} {len(faces)} 0\n")
        for v in unique_verts:
            f.write(f"{v[0].item()} {v[1].item()} {v[2].item()}\n")
        for i, face in enumerate(faces):
            r, g, b = colors[i].tolist()
            f.write(f"3 {face[0].item()} {face[1].item()} {face[2].item()} {r} {g} {b} 255\n")

    print(f"saved {args.output_name}")

if __name__ == "__main__":
    main()