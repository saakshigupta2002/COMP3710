#!/bin/bash 
#SBATCH --job-name=saakshi_CIFAR
#SBATCH --mail-type=All
#SBATCH --mail-user=saakshi.gupta@uqconnect.edu.au
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1
#SBATCH -o test_ou.txt
#SBATCH -e test_er.txt

conda activate conda-env

python ~/task2.py

conda deactivate
