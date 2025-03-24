# ======================================================================
# Author: Austin Tao (modified from drug_experiment/merge_eval.py)
#
# Script to determine why we are seeing higher than average test loss 
# values with our merge.
#======================================================================

import re
import torch
import numpy as np
import pandas as pd
import evaluate
from typing import Any, Tuple, Dict  # for type annotations
from torch.nn.functional import softmax
from datasets import load_dataset,Dataset
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, PreTrainedTokenizerBase
from sklearn.metrics import accuracy_score


# define constants

MODEL_PATHS: list[str] = [
    #"gpt2_untrained",
    "gpt2_f_experiment_0",
    "gpt2_f_experiment_1",
    "gpt2_f_experiment_2",
    "gpt2_f_experiment_3",
    "gpt2_f_experiment_4",
    "gpt2_f_experiment_5",
    "gpt2_f_experiment_6",
    "gpt2_f_experiment_7",
    "gpt2_f_experiment_8",
    "gpt2_f_experiment_9",
    "gpt_f_experiment_large",
    "gpt2_m_experiment_dare_linear",
    "gpt2_m_experiment_ties",
    "gpt2_m_experiment_linear",
    "gpt2_m_experiment"       # note that this is DARE TIES
] 

# get the paths for drug dataset models
DRUG_MODEL_PATHS: list[str] = [
    "gpt2_f_experiment_0_drug_data",
    "gpt2_f_experiment_1_drug_data",
    "gpt2_f_experiment_2_drug_data",
    "gpt2_f_experiment_3_drug_data",
    "gpt2_f_experiment_4_drug_data",
    "gpt2_f_experiment_drug_data_large",
    "gpt2_m_experiment_drug_data_linear",
    "gpt2_m_experiment_drug_data_ties",
    "gpt2_m_experiment_drug_data_dare_linear",
    "gpt2_m_experiment_drug_data_dare_ties",
    "gpt2_m_experiment_drug_data_dare_linear_test",
    "gpt2_m_experiment_drug_data_ties_test"
]

DRUG_REVIEW_DATASET_PATH: str = "lewtun/drug-reviews"
YELP_REVIEW_DATASET_PATH: str = "yelp_review_full"
PROB_CSV_FILE_NAME: str = "{}/{}_{}_probabilities.csv"
ACC_FILE_NAME: str = "{}/{}_accuracy.txt"
TEST_LOSS_EXP_DIR: str = "test_loss_exp"
RESULTS_FILE_NAME: str = "model_results.txt"
EMPTY_STRING: str = ""
SEED: int = 467
TOKENIZE_MAX_LENGTH: int = 1024
APPEND_MODE: str = APPEND_MODE
WRITE_MODE: str = "w"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
DEFAULT_FILE_PREFIX_EXP: str = r"\/(.*)"
TEXT_KEY: str = "text"
LABEL_KEY: str = "label"
TRUE_LABEL_COL: str = "true_label"


def eval_with_probs(trainer: Trainer, test_dataset: Dataset, validation_dataset:Dataset, file_prefix: str,
                    output_dir: str = TEST_LOSS_EXP_DIR) -> None:
    """
    Evaluate the model on test and validation datasets using the provided Trainer object,
    compute probabilities, and save the results along with accuracy to CSV files.

    Args:
        - trainer (Trainer): The Hugging Face Trainer object.
        - test_dataset (datasets.Dataset): The dataset for testing.
        - validation_dataset (datasets.Dataset): The dataset for validation.
        - file_prefix (str): prefix to add to the outputted file
        - output_dir (str): Directory path where the results will be saved.

    Returns:
        - None
    """
    # define datasets
    datasets = {"test": test_dataset, "validation": validation_dataset}

    # go through both test and val sets
    for name, dataset in datasets.items():
        # Predict using the trainer
        predictions = trainer.predict(dataset)
        
        # Extract logits and convert to probabilities
        logits = predictions.predictions
        probabilities = softmax(torch.tensor(logits), dim=-1).numpy()

        # add the true labels
        labels = dataset[LABEL_KEY]
        df = pd.DataFrame(probabilities)
        df[TRUE_LABEL_COL] = dataset[LABEL_KEY]
        
        # Save probabilities to a CSV file
        probabilities_file = EMPTY_STRING.format(output_dir, file_prefix, name)
        df.to_csv(probabilities_file, index = False)
        print(f"Saved {name} probabilities to {probabilities_file}")
        
        # Calculate and print accuracy if labels are present
        if hasattr(predictions, "label_ids") and predictions.label_ids is not None:
            true_labels = predictions.label_ids
            predicted_labels = logits.argmax(axis=-1)
            accuracy = accuracy_score(true_labels, predicted_labels)
            print(f"{name.capitalize()} Accuracy: {accuracy}")

            # Save accuracy to a file
            with open(ACC_FILE_NAME.format(output_dir, name), WRITE_MODE) as f:
                f.write(f"{name.capitalize()} Accuracy: {accuracy}\n")



def train_test_val_split(dataset, tokenizer, prop_eval: float = 0.2, _DRUG_DATA = False):
    """
    Method to split dataset into training, testing, and validation (and tokenized)

    Args:
        - dataset:
        - tokenizer: 
        - prop_eval: proportion WITHIN testing data to set aside for validation

    Returns:
        - Tuple containing training, testing, and validation datasets (tokenized)
    """
    if _DRUG_DATA:
        # separate into training and eval
        train_data = dataset["train"]
        eval_data = dataset["test"]

        # separagte based on reviews and ratings
        reviews_train = train_data["review"]
        ratings_train = train_data["rating"]
        reviews_eval = eval_data["review"]
        ratings_eval = eval_data["rating"]

        # create datasets from dictionaries
        drug_data_train = {"label": [int(rating-1) for rating in ratings_train],TEXT_KEY: reviews_train}
        drug_data_train = Dataset.from_dict(drug_data_train)
        drug_data_eval = {"label": [int(rating-1) for rating in ratings_eval],TEXT_KEY: reviews_eval}
        drug_data_eval = Dataset.from_dict(drug_data_eval)

        # tokenization
        tokenized_drug_train = drug_data_train.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024), batched=True)
        tokenized_drug_test = drug_data_eval.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024), batched=True)
        eval_dataset = tokenized_drug_test.shuffle(seed=SEED)

        # Define the size of the validation set (e.g., 20% of the total data)
        validation_size = int(0.2 * len(eval_dataset))
        indices = list(range(len(eval_dataset)))
        test_indices = indices[validation_size:]
        validation_indices = indices[:validation_size]

        # Split the shuffled training dataset into training and validation sets
        test_dataset = eval_dataset.select(test_indices)
        validation_dataset = eval_dataset.select(validation_indices)

        # rename training set for continuity
        train_dataset = tokenized_drug_train

    else:
        # tokenization
        tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024),
                                                   batched=True)
        train_dataset = tokenized_yelp["train"]
        eval_dataset = tokenized_yelp["test"]

        # Define the size of the validation set (e.g., 20% of the total data)
        validation_size = int(0.2 * len(eval_dataset))
        indices = list(range(len(eval_dataset)))
        test_indices = indices[validation_size:]
        validation_indices = indices[:validation_size]

        # Split the shuffled training dataset into training and validation sets
        test_dataset = eval_dataset.select(test_indices)
        validation_dataset = eval_dataset.select(validation_indices)


    return train_dataset, test_dataset, validation_dataset



def write_results(test_results: Any, val_results: Any, file_path: str = RESULTS_FILE_NAME) -> None:
    """
    Write the results of our model, including the current date and time, test results, and validation results,
    to a specified file. The file will be appended to if it already exists.
    
    Args:
        - test_results (Any): The results from testing the model. This can be any data type that is compatible
                              with string formatting, typically str or a type that implements __str__.
        - val_results (Any): The results from validating the model. This can be any data type that is compatible
                             with string formatting.
        - file_path (str, optional): The path to the file where results will be written. Defaults to "model_results.txt".
    
    Returns:
        - None: This function does not return any values.
    """
    # Hold the current date and time information
    time_now = datetime.today().strftime(DATE_FORMAT)

    # Generate result string
    result_str = "Date and Time: {}\nTest Results: {}\nValidation Results: {}".format(time_now, test_results, val_results)

    # Write to file
    with open(file_path, APPEND_MODE) as file:
        file.write(result_str)


def preprocess_function(examples: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, _MAX_TOKENIZER_LEN: int = TOKENIZE_MAX_LENGTH) -> Dict[str, Any]:
    """
    Preprocesses the text data by tokenizing the input text using the specified tokenizer.
    
    Args:
        - examples (Dict[str, Any]): dict containing the text examples with keys corresponding to feature names and values.
        - tokenizer (PreTrainedTokenizerBase): The tokenizer instance from the transformers library

    Returns:
        - Dict[str, Any]: A dictionary containing the tokenized text data.
    """
    return tokenizer(examples["text"], truncation = True, max_length = _MAX_TOKENIZER_LEN)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Computes the accuracy of model predictions using the "accuracy" metric from the Hugging Face datasets library.
    
    Args:
        - eval_pred (Tuple[np.ndarray, np.ndarray]): A tuple containing two elements:
            1. predictions (np.ndarray): The raw model output logits or probabilities.
            2. labels (np.ndarray): The true labels against which to evaluate the predictions.
    
    Returns:
        - Dict[str, float]: A dictionary containing the computed accuracy metric.
    """
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis = 1)
    return accuracy.compute(predictions = predictions, references = labels)


def run_exp(model_path: str, dataset_path: str, file_prefix: str = EMPTY_STRING,
            _DEFAULT_FILE_PREFIX_EXP: str = DEFAULT_FILE_PREFIX_EXP, _DRUG_DATA = False) -> None:
    """
    Run and get the experiment probabilities for a particular model

    Args:
        - model_path (str): path to the model in HF
        - dataset_path (str): path to the dataset in HF

    Returns:
        None. Writes to file if successful.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # load dataset
    dataset = load_dataset(dataset_path)
    
    # split into train-test-val
    training_dataset, test_dataset, validation_dataset = train_test_val_split(dataset, tokenizer, _DRUG_DATA = _DRUG_DATA)

    # Load model directly
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # train model
    trainer = Trainer(
        model=model,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # set file prefix to model name if not provided
    if file_prefix == EMPTY_STRING:
        match = re.search(_DEFAULT_FILE_PREFIX_EXP, model_path)
        file_prefix = match.group(1)

    # get probabilities and write to file
    eval_with_probs(trainer, test_dataset, validation_dataset, file_prefix)


def main() -> None:

    # get the path of all the models for yelp and imdb

    # run and get prediction probabilities and accuracies for yelp
    for path in model_paths:
        run_exp(path, YELP_REVIEW_DATASET_PATH)

    for path in drug_model_paths:
        run_exp(path, DRUG_REVIEW_DATASET_PATH, _DRUG_DATA = True)


if __name__ == "__main__":
    main()


# EOF
