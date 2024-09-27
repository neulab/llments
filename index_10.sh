#!/bin/bash
#SBATCH --job-name=index10
#SBATCH --mem=90G
#SBATCH --output=idx10.log
#SBATCH --partition=general
#SBATCH --gres=gpu:L40:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

python index_code_10.py