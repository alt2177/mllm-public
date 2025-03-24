#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning # Job name
#SBATCH --nodes=1                      # Use one node
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:1                   # Request one GPU
#SBATCH --output=output.log     # Standard output and error log
#SBATCH --mail-type=END,FAIL           # Send email on job END and FAIL
#SBATCH --main-user=[YOUR_EMAIL]

# run script via Makefile
#pip uninstall transformers
#git clone https://github.com/cg123/mergekit.git
#cd mergekit
#pip install -e .
#cd ..
python merge_experiment.py
