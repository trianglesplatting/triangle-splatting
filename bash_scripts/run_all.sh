# tandt
sbatch bash_scripts/tandt/eval_truck.sh $1
sbatch bash_scripts/tandt/eval_train.sh $1

# m360 indoor
sbatch bash_scripts/m360_indoor/eval_room.sh $1
sbatch bash_scripts/m360_indoor/eval_counter.sh $1
sbatch bash_scripts/m360_indoor/eval_kitchen.sh $1
sbatch bash_scripts/m360_indoor/eval_bonsai.sh $1

# m360 outdoor
sbatch bash_scripts/m360_outdoor/eval_bicycle.sh $1
sbatch bash_scripts/m360_outdoor/eval_flowers.sh $1
sbatch bash_scripts/m360_outdoor/eval_garden.sh $1
sbatch bash_scripts/m360_outdoor/eval_stump.sh $1
sbatch bash_scripts/m360_outdoor/eval_treehill.sh $1