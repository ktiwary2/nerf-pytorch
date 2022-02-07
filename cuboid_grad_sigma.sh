#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1

# Loading the required module
source /etc/profile
#module load anaconda/2021a #cuda/11.1
source activate nerf_pl

# Run the script
python3 run_sm_nerf.py --config ./configs/cuboid_grad_light_sigma.txt