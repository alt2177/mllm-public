#!/bin/bash

## Resource Requests

# Function to handle preemption
preemption_handler() {
    echo "Job preempted, exiting..."
    exit 143  # Exit code for SIGTERM
}

# Trap SIGUSR1 and call preemption handler
trap preemption_handler SIGUSR1

# Create a Python virtual environment and install dependencies
echo "Setting up Python environment..."
module load python/3.8
module load cuda/10.1
python -m venv mllm_venv
source mllm_venv/bin/activate
pip install --upgrade pip
pip install transformers torch datasets evaluate
pip install -r requirements.txt

# Run the main job script
# Make sure to edit this to be the module you want to run!
# Example: python -m src.{MODULE_DIR}.{MODULE_NAME}
python -m experiment_configs.eval_scripts.eval_gpt2_linear_merge

# Clean up environment
echo "Cleaning up the environment..."
deactivate
rm -rf mllm_venv

# Additional clean-up commands (adjust as necessary)
echo "Additional cleanup..."
rm -rf __pycache__
rm -rf src/__pycache__
rm -rf tests/__pycache__
find . -name '*.pyc' -exec rm -f {} +
find . -name '*.pyo' -exec rm -f {} +
find . -name '*~' -exec rm -f {} +
# find . -type d -name "mergekit" -exec rm -rf {} +
# find . -type d -name "tmp_trainer" -exec rm -rf {} +
# find . -type d -name "test_MLLM" -exec rm -rf {} +
