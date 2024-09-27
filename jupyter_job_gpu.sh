#!/bin/bash
#SBATCH --job-name=mihirgpu
#SBATCH --mem=500G
#SBATCH --error=jupyter_gpu_1.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888
