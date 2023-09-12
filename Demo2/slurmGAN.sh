#!/bin/bash 
#SBATCH --job-name=saakshi_GAN
#SBATCH --mail-type=All
#SBATCH --mail-user=saakshi.gupta@uqconnect.edu.au
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1
#SBATCH -o test_ou.txt
#SBATCH -e test_er.txt

conda activate conda-env

python ~/task5.py

conda deactivate
