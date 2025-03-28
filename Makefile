# Makefile (generated by ChatGPT)

# Specify the Python interpreter
PYTHON = python3
VENV = venv

# Create a virtual environment
venv: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	${VENV}/bin/pip install -Ur requirements.txt
	touch $(VENV)/bin/activate

# Install dependencies
install: venv

# Run your program
run:
	sbatch lambda_fine_tune.sh
	make clean


# Run tests
test: venv
	${VENV}/bin/pytest tests/

# Clean up
# TODO: find a way to set up deletion of output_dir in an MLLM instance
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.log' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -type d -name "mergekit" -exec rm -rf {} +
	find . -type d -name "tmp_trainer" -exec rm -rf {} +
	find . -type d -name "test_MLLM" -exec rm -rf {} +

# Clean everything including virtualenv
clean-all: clean
	rm -rf $(VENV)

.PHONY: install run test clean clean-all

# EOF