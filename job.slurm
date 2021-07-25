#!/bin/bash
#SBATCH -o job%J.out
#SBATCH --partition=gpu_part
#SBATCH --qos=low
#SBATCH -J myFirstJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -w compute1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#srun hostname


source deactivate
source activate /public/tools/conda/envs/pytorch_env

#jupyter nbconvert --execute --clear-output train.ipynb
python train.py
