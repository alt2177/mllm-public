from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np
import torch
from torch.nn.functional import softmax 

# from mllm.data.load_drug_data import load_drug_data
# from mllm.core.MLLM import MLLM 
# from mergekit.config import MergeConfiguration
# from mergekit.merge import MergeOptions, run_merge

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,max_length=1024)

def main():
    # set our collective token
    access_token = HF_ACCESS_TOKEN 
    username = HF_USERNAME
    output_repo = "gpt2_finetune_2_with_prob_epoch_5"

    # load and tokenize data
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_imdb = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)

    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # create model
    model = AutoModelForSequenceClassification.from_pretrained(
        "openai-community/gpt2",
        num_labels=2,
        use_auth_token = access_token,
        id2label=id2label, 
        label2id=label2id
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # create training-val split
    train_dataset = tokenized_imdb["train"].shuffle(seed=42)
    eval_dataset = tokenized_imdb["test"].shuffle(seed=42)
    #small_train_dataset = tokenized_imdb["train"].shuffle(seed=42).select(range(1000))
    #small_eval_dataset = tokenized_imdb["test"].shuffle(seed=42).select(range(1000))
 
    output_dir = "imdb_finetune_gpt2_test_epoch_5_with_prob_1_merge_output"
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
	hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    result = trainer.train()
    print(result)
    predictions = trainer.predict(eval_dataset)
    logits = predictions.predictions
    probs  = softmax(torch.tensor(logits),dim=1).numpy()
    torch.save(probs,"test_probs_epoch_5.pt")
    
#    trainer.push_to_hub(f"{username}/{output_repo}")
	
    HfFolder.save_token(access_token)

# Optionally, you can directly use the token with the HfApi instance
    api = HfApi()
    user = api.whoami(token=access_token)
    print("Logged in as:", user['name'])
    
    print('UPLOADING')
    api.upload_folder(
        folder_path=f"./{output_dir}",
        repo_id=f"{username}/{output_repo}",
        repo_type="model"
    )
    print('uploading done!')


    # model eval
    #eval_result = trainer.evaluate(eval_dataset=tokenized_imdb["test"])


    #print(f"Fine tune model accuracy : {eval_result['eval_accuracy']}")



if __name__ == "__main__":
    main()
