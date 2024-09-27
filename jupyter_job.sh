#!/bin/bash
#SBATCH --job-name=notebookmi
#SBATCH --mem=30G
#SBATCH --error=jupyter.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888
