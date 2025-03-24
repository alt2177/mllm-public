#!/bin/bash

## Resource Requests
#SBATCH --nodes=1# Use one node
#SBATCH --gres=gpu:8                   # Request one GPU
#SBATCH --output=output.log     # Standard output and error log

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
python main.py

