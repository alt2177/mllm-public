#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning # Job name
#SBATCH --nodes=1                      # Use one node
#SBATCH --partition= # Specify the partition to run on
#SBATCH --nodelist=saruman
#SBATCH --gres=gpu:1                   # Request one GPU
#SBATCH --output=output.log     # Standard output and error log
#SBATCH --mail-type=END,FAIL           # Send email on job END and FAIL
#SBATCH --mail-user=
#SBATCH --output=my_gpu_job_%j.log     # Standard output and error log
#SBATCH --mail-type=BEGIN, END, FAIL           # Send email on job END and FAIL

# Load any modules and set up your environment
module load python/3.8
module load cuda/10.1                 # Load CUDA module, if required

# run script via Makefile
pip uninstall transformers
pip install transformers==4.28.0
pip install torch
pip install datasets
pip install evaluate
pip install numpy
pip install huggingface
pip install regex --upgrade
pip install huggingface_hub

echo $(pwd)

python3 ./main.py
make run