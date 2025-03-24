import torch
import numpy as np
import yaml 
from pathlib import Path
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi

# contains shared utils and constants
from utils import load_yaml_config, get_config_section, ACCESS_TOKEN, CONFIG_DIR_PATH, SEED, compute_metrics


CONFIG_YML = CONFIG_DIR_PATH / 'example_config.yml'     # change this to be your config file
COPY_TOKENIZER = True 
LAZY_UNPICKLE = False  
LOW_CPU_MEMORY = False 


# load your config
config = load_yaml_config(CONFIG_YML, 'r')

# get specific sections of the config
DATASET_CONFIG = get_config_section(config, "dataset")
MODEL_CONFIG = get_config_section(config, "model")
MERGEKIT_CONFIG = get_config_section(config, "mergekit_config")
TRAINING_ARGS = get_config_section(config, "training_args")
HF_ARGS = get_config_section(config, "hugging_face")



if __name__ == "__main__":
    np.random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load and tokenize data
    dataset = load_dataset(DATASET_CONFIG["name"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["name"], use_auth_token = ACCESS_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_imdb = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,
                                                           max_length=1024), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # create train-val split
    train_dataset = tokenized_imdb["train"].shuffle(seed=42)
    eval_dataset = tokenized_imdb["test"].shuffle(seed=42)

    with torch.no_grad():

        # define a new trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(TRAINING_ARGS),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # train
        result = trainer.train()
        print(result)
        predictions = trainer.predict(eval_dataset)
        logits = predictions.predictions
        probs  = softmax(torch.tensor(logits),dim=1).numpy()
        torch.save(probs,"test_probs_epoch_1.pt")

        # upload to HF if specified
        if HF_ARGS["upload_to_hf"]:
            api = HfApi()
            user = api.whoami(token=ACCESS_TOKEN)
            print("Logged in as:", user['name'])
            
            print('UPLOADING')
            api.upload_folder(
                folder_path=f"./{HF_ARGS["hf_output_dir"]}",
                repo_id=f"{HF_ARGS["hf_username"]}/{HF_ARGS["hf_output_dir"]}",
                repo_type="model"
            )
            print('uploading done!')
