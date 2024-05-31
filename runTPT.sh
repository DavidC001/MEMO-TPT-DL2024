#!/bin/sh
#SBATCH -p edu-20h
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1024M
#SBATCH -N 1
#SBATCH -t 0-20:00

module load cuda/12.1

source /home/davide.cavicchini/.bashrc
conda activate SIV_hpe

python3 EasyTPT/tests.py

