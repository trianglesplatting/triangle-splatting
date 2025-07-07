import argparse
import base64
import torch
import numpy as np
from utils.sh_utils import eval_sh
from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Attributes, Scene, Node



def main():
    parser = argparse.ArgumentParser(description="Convert a checkpoint to a colored GLB file.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the input checkpoint file (e.g., point_cloud_state_dict.pt)")
    parser.add_argument("--output_name", type=str, help="Name of the output GLB file (e.g., mesh_colored.glb)")
    args = parser.parse_args()

    sd = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)

    points = sd["triangles_points"]
    features_dc = sd["features_dc"]
    features_rest = sd["features_rest"]
    opacity =  torch.clamp(1 / (1 + torch.exp(-sd["opacity"])), 0.0, 1.0)
    # opacity = torch.clamp(sd["opacity"], 0.0, 1.0)

    features = torch.cat((features_dc, features_rest), dim=1)

    num_coeffs = features.shape[1]
    max_sh_degree = int(np.sqrt(num_coeffs) - 1)

    shs = features.permute(0, 2, 1).contiguous()

    centroids = points.mean(dim=1)
    camera_center = torch.zeros(3)
    dirs = centroids - camera_center
    dirs_norm = dirs / dirs.norm(dim=1, keepdim=True)

    # rgb = eval_sh(max_sh_degree, shs, dirs_norm)
    rgb = eval_sh(0, shs, dirs_norm)

    colors_f = torch.clamp(rgb + 0.5, 0.0, 1.0)
    colors = torch.concat([
      colors_f.reshape(-1, 3),
      opacity,
    ], dim=1).to(torch.float32).detach().numpy().astype(np.float32)
    n_channels = colors.shape[1]

    # Prepare vertices and faces
    num_triangles = points.shape[0]
    all_verts = points.reshape(-1, 3)  # Shape: (num_triangles * 3, 3)
    face_indices = torch.arange(num_triangles * 3).reshape(num_triangles, 3)

    vertex_colors = colors.reshape((-1, 1, n_channels))
    vertex_colors = np.tile(vertex_colors, (1, 3, 1))
    vertex_colors = vertex_colors.reshape((-1, n_channels))

    all_verts = all_verts.reshape(-1, 3).detach().numpy().astype(np.float32)
    all_colors = vertex_colors
    all_indices = face_indices.reshape(-1).detach().numpy().astype(np.uint32)

    # Create GLTF2 object
    gltf = GLTF2()

    # 1. Create buffers
    buffer = Buffer(byteLength=len(all_verts.tobytes()) + len(all_colors.tobytes()) + len(all_indices.tobytes()))

    # BufferView for positions (ARRAY_BUFFER = 34962)
    buffer_view_positions = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(all_verts.tobytes()),
        target=34962  # ✅ Numeric value
    )

    # BufferView for colors (ARRAY_BUFFER = 34962)
    buffer_view_colors = BufferView(
        buffer=0,
        byteOffset=len(all_verts.tobytes()),
        byteLength=len(all_colors.tobytes()),
        target=34962  # ✅ Numeric value
    )

    # BufferView for indices (ELEMENT_ARRAY_BUFFER = 34963)
    buffer_view_indices = BufferView(
        buffer=0,
        byteOffset=len(all_verts.tobytes()) + len(all_colors.tobytes()),
        byteLength=len(all_indices.tobytes()),
        target=34963  # ✅ Numeric value
    )

    # Accessor for positions (FLOAT = 5126)
    accessor_positions = Accessor(
        bufferView=0,
        componentType=5126,  # ✅ Numeric value
        count=len(all_verts),
        type="VEC3",
        max=all_verts.max(axis=0).tolist(),
        min=all_verts.min(axis=0).tolist()
    )

    # Accessor for colors (UNSIGNED_BYTE = 5121)
    accessor_colors = Accessor(
        bufferView=1,
        componentType=5126,  # ✅ Numeric value
        count=len(all_colors),
        type="VEC4",
        max=all_colors.max(axis=0).tolist(),
        min=all_colors.min(axis=0).tolist()
    )

    # Accessor for indices (UNSIGNED_SHORT = 5123)
    accessor_indices = Accessor(
        bufferView=2,
        componentType=5125,  # ✅ Numeric value
        count=int(len(all_indices)),
        type="SCALAR",
        max=[int(all_indices.max())],
        min=[int(all_indices.min())]
    )

    # 4. Create mesh primitive
    primitive = Primitive(
        attributes=Attributes(POSITION=0, COLOR_0=1),
        indices=2,
        mode=4,
    )

    mesh = Mesh(primitives=[primitive])

    # 5. Create scene and node
    scene = Scene(nodes=[0])  # Reference the node index
    node = Node(mesh=0)  # Reference the mesh index

    # 6. Add to GLTF2
    gltf.buffers.append(buffer)
    gltf.bufferViews.extend([buffer_view_positions, buffer_view_colors, buffer_view_indices])
    gltf.accessors.extend([accessor_positions, accessor_colors, accessor_indices])
    gltf.meshes.append(mesh)
    gltf.nodes.append(node)
    gltf.scenes.append(scene)

    # 6. Set binary data
    gltf.set_binary_blob(
        all_verts.tobytes() + all_colors.tobytes() + all_indices.tobytes()
    )

    # Save to .gltf
    gltf.save(args.output_name)

    print(f"saved {args.output_name}")

if __name__ == "__main__":
    main()