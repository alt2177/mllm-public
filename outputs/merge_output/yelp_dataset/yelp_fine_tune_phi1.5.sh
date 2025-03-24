#!/bin/bash

## Resource Requests
#SBATCH --job-name=fine_tuning_yelp_phi1.5 # Job name
#SBATCH --nodes=1# Use one node
#SBATCH --account=[YOUR_ACCOUNT]
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:8                  # Request one GPU
#SBATCH --output=output_3.log     # Standard output and error log
#SBATCH --mail-type=BEGIN,FAIL,END           # Send email on job END and FAIL
#SBATCH --main-user=[YOUR_EMAIL] 

# run script via Makefile
#pip uninstall transformers
#pip install transformers==4.34.0
#pip install torch 
#pip install datasets
#pip install evaluate
#pip install numpy
#pip uninstall huggingface_hub
#pip install huggingface_hub==0.19.4
#pip install regex --upgrade
#pip install huggingface_hub
#git config user.email "[YOUR EMAIL HERE]"
#git config user.name "[YOUR GIT USERNAME]"

#python -m torch.distributed.launch --nproc_per_node=8 main.py
python main_2.py
