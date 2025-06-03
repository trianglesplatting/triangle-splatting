#!/bin/bash
#SBATCH --output=logs/eval_truck.log
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=30G # Memory to allocate in MB per allocated CPU core
#SBATCH --time="0-00:40:00" # Max execution time


micromamba activate triangle_splatting

python train.py \
-s /gpfs/scratch/acad/telim/datasets/tandt/truck/  \
-m models/$1/truck \
--quiet \
--eval \
--max_shapes 2000000 \
--outdoor \


python render.py --iteration 30000 -s /gpfs/scratch/acad/telim/datasets/tandt/truck/ -m models/$1/truck --eval --skip_train --quiet

python metrics.py -m models/$1/truck