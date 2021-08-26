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
source activate $torch_env

#jupyter nbconvert --execute --clear-output train.ipynb
#python train_3D.py
#python test.py
#python 1.py
python predict.py
