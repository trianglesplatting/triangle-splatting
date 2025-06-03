#!/bin/bash
#SBATCH --output=logs/eval_garden.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=30G # Memory to allocate in MB per allocated CPU core
#SBATCH --time="0-00:55:00" # Max execution time


micromamba activate triangle_splatting

python train.py \
-s /gpfs/scratch/acad/telim/datasets/MipNeRF360/garden  \
-i images_4 \
-m models/$1/garden \
--quiet \
--eval \
--max_shapes 5200000 \
--outdoor \


python render.py --iteration 30000 -s /gpfs/scratch/acad/telim/datasets/MipNeRF360/garden -m models/$1/garden --eval --skip_train --quiet

python metrics.py -m models/$1/garden