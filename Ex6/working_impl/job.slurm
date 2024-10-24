#!/bin/sh
#SBATCH --partition=TDT4200
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --job-name="2D Wave Equation"
#SBATCH --output=out.txt
./parallel
