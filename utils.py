"""
Author: Austin
Shared utils that will be used across experiments and configurations
"""

def load_yaml_config(config_path: Any, mode: str = 'r', encoding: str = "utf-8"):
    """
    Load YAML configuration file.

    @param config_path: the path to the yaml config
    @param mode: the mode in which to open the file
    @param encoding: the desired encoding
    @returns yaml file pointer
    """
    with open(config_path, mode, encoding=encoding) as file:
        return yaml.safe_load(file)


def get_config_section(config, section_name: str):
    """
    Get a specific section from the configuration.
    """
    try:
        section = config['config'][section_name]
        return section
    except KeyError:
        raise ValueError(f"Section '{section_name}' not found in the configuration.")


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,max_length=1024)

# EOF