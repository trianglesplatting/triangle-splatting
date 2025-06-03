import os
from argparse import ArgumentParser

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/dtu")
parser.add_argument('--dtu', "-dtu", required=True, type=str)

parser.add_argument('--max_shapes', default=500000, type=int)
parser.add_argument('--lambda_normals', default=0.0028, type=float)
parser.add_argument('--lambda_dist', default=0.014, type=float)
parser.add_argument('--iteration_mesh', default=25000, type=int)
parser.add_argument('--densify_until_iter', default=25000, type=int)
parser.add_argument('--lambda_opacity', default=0.0044, type=float)
parser.add_argument('--importance_threshold', default=0.027, type=float)
parser.add_argument('--lr_triangles_points_init', default=0.0015, type=float)


args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args = (
        f" --test_iterations -1 --depth_ratio 1.0 -r 2 --eval --max_shapes {args.max_shapes}"
        f" --lambda_normals {args.lambda_normals}"
        f" --lambda_dist {args.lambda_dist}"
        f" --iteration_mesh {args.iteration_mesh}"
        f" --densify_until_iter {args.densify_until_iter}"
        f" --lambda_opacity {args.lambda_opacity}"
        f" --importance_threshold {args.importance_threshold}"
        f" --lr_triangles_points_init {args.lr_triangles_points_init}"
        f" --lambda_size {0.0}"
        f" --no_dome"
    )

    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)


if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python mesh.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python mesh.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)


if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for scene in dtu_scenes:
        scan_id = scene[4:]
        ply_file = f"{args.output_path}/{scene}/train/ours_30000/"
        iteration = 30000
        output_dir = f"{args.output_path}/{scene}/"
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {args.output_path}/{scene}/train/ours_30000/fuse_post.ply " + \
            f"--scan_id {scan_id} --output_dir {output_dir}/ " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        print(string)
        os.system(string)


import json

average = 0
for scene in dtu_scenes:
    output_dir = f"{args.output_path}/{scene}/"
    with open(output_dir + '/results.json', 'r') as f:
        results = json.load(f)
    print("Results: ", results)
    average += results['overall']
average /= len(dtu_scenes)
