#!/bin/bash
#SBATCH --output=logs/eval_bonsai.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=30G # Memory to allocate in MB per allocated CPU core
#SBATCH --time="0-00:50:00" # Max execution time


micromamba activate triangle_splatting

python train.py \
-s /gpfs/scratch/acad/telim/datasets/MipNeRF360/bonsai  \
-i images_2 \
-m models/$1/bonsai \
--quiet \
--eval \
--max_shapes 3000000 \
--importance_threshold 0.025 \
--lr_sigma 0.0008 \
--opacity_lr 0.014 \
--lambda_normals 0.00004 \
--lambda_dist 1 \
--iteration_mesh 5000 \
--lambda_opacity 0.0055 \
--lambda_dssim 0.4 \
--lr_triangles_points_init 0.0015 \
--lambda_size 5e-8 


python render.py --iteration 30000 -s /gpfs/scratch/acad/telim/datasets/MipNeRF360/bonsai -m models/$1/bonsai --eval --skip_train --quiet

python metrics.py -m models/$1/bonsai