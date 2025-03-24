#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning_billsum_t5 # Job name
#SBATCH --nodes=1# Use one node
#SBATCH --account=[YOUR_ACCOUNT]
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:6                  # Request one GPU
#SBATCH --output=output.log     # Standard output and error log
#SBATCH --mail-type=FAIL,END           # Send email on job END and FAIL
#SBATCH --main-user=[YOUR_EMAIL] 

# run script via Makefile
#pip uninstall transformers
#pip install transformers==4.28.0
#pip install torch 
#pip install datasets
#pip install evaluate
#pip install numpy
#pip install huggingface
#pip install regex --upgrade
#pip install huggingface_hub
#git config user.email "[YOUR EMAIL HERE]"
#git config user.name "[YOUR GIT USERNAME]"
#pip install accelerate -U
#python -m torch.distributed.launch --nproc_per_node=6 main.py
python t5.py
