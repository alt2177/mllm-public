#!/bin/bash

## Resource Requests
#SBATCH --job-name=test_loss_exp# Job name
#SBATCH --nodes=1# Use one node
#SBATCH --account=[YOUR_ACCOUNT]
#SBATCH --partition=lambda        # Specify the partition to run on
#SBATCH --gres=gpu:2                   # Request one GPU
#SBATCH --output=output_test_loss_exp.log     # Standard output and error log
#SBATCH --mail-type=END,FAIL           # Send email on job END and FAIL
#SBATCH --main-user=[YOUR_EMAIL]

# function to requeue jobs if preempted
requeue_job() {
    echo "Job $SLURM_JOB_ID is being requeued..."
        sbatch $0
}

# Setup a trap for the SIGUSR1 signal which Slurm uses for preemption
trap 'requeue_job' SIGUSR1

# run script via Makefile
pip uninstall transformers
pip install transformers==4.28.0
pip install accelerate -U
pip install transformers
pip install torch
pip install datasets
pip install evaluate
pip install numpy
pip install huggingface
pip install regex --upgrade
pip install huggingface_hub
git config user.email "[YOUR EMAIL HERE]"
git config user.name "[YOUR GIT USERNAME]"
git remote set-url origin https://huggingface.co/yelp_finetuned_6gpu_full
python3 test_loss_exp.py
