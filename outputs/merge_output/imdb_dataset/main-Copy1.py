from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
<<<<<<< HEAD
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
=======
from datasets import load_dataset
from transformers import AutoTokenizer
>>>>>>> main
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

<<<<<<< HEAD
=======
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,max_length=1024)

>>>>>>> main
def main():
    # set our collective token
    access_token = HF_ACCESS_TOKEN 
    username = HF_USERNAME
<<<<<<< HEAD
<<<<<<< HEAD:merge_output/imdb_dataset/main-Copy1.py
    output_repo = "gpt2_finetune_2_with_prob"
=======
    output_repo = "yelp_finetuned_sbatch_upload_4gpu_BIG"
>>>>>>> 4f63ef5a3 (main is working well with pushing):main.py
=======
    output_repo = "gpt2_finetune_2_with_prob"
>>>>>>> main

    # load and tokenize data
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

<<<<<<< HEAD
<<<<<<< HEAD:merge_output/imdb_dataset/main-Copy1.py
=======
>>>>>>> main
    tokenized_imdb = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)

    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

<<<<<<< HEAD
=======
    tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)
    
    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

>>>>>>> 4f63ef5a3 (main is working well with pushing):main.py
=======
>>>>>>> main
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

<<<<<<< HEAD
<<<<<<< HEAD:merge_output/imdb_dataset/main-Copy1.py
=======
>>>>>>> main
    # create training-val split
    small_train_dataset = tokenized_imdb["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_imdb["test"].shuffle(seed=42).select(range(1000))
 
    output_dir = "imdb_finetune_gpt2_test_epoch_5_with_prob_1"
<<<<<<< HEAD
=======
    small_train_dataset = tokenized_yelp["train"].shuffle(seed=42).select(range(300000))
    small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42).select(range(50000))

    HfFolder.save_token(access_token)

    api = HfApi()
    user = api.whoami(token=access_token)
    try:
    	create_repo(f"{username}/{output_repo}", repo_type="model")
    except:
        print('error creating repo for model. it probably already exists')

    output_dir = "yelp_finetune_gpt2_test"
>>>>>>> 4f63ef5a3 (main is working well with pushing):main.py
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=6e-5,    #read that with larger batch size we can increase learning rate
#	gradient_accumulation_steps=4,   #want to explore what this is
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=1,
	fp16=True,
=======
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
>>>>>>> main
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
<<<<<<< HEAD
	hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token,
	push_to_hub=True
=======
        push_to_hub=True,
	hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token
>>>>>>> main
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
<<<<<<< HEAD
        compute_metrics=compute_metrics,
    )
    print('************TRAINING STARTED*************')
    trainer.train()
    print('*************TRAINING COMPLETED**************')
   # trainer.evaluate()

<<<<<<< HEAD:merge_output/imdb_dataset/main-Copy1.py
=======
        compute_metrics=compute_metrics
    )

>>>>>>> main
    result = trainer.train()
    print(result)
    predictions = trainer.predict(small_eval_dataset)
    logits = predictions.predictions
    probs  = softmax(torch.tensor(logits),dim=1).numpy()
    torch.save(probs,"test_probs_2.pt")
    
#    trainer.push_to_hub(f"{username}/{output_repo}")
	
    HfFolder.save_token(access_token)
<<<<<<< HEAD
=======
    #HfFolder.save_token(access_token)
>>>>>>> 4f63ef5a3 (main is working well with pushing):main.py

    #api = HfApi()
   # user = api.whoami(token=access_token)
    #print("Logged in as:", user['name'])

    #create_repo(f"{username}/{output_repo}", repo_type="model")

    #api.upload_folder(
    #	folder_path=f"./{output_dir}",
#	repo_id=f"{username}/{output_repo}",
 #       repo_type="model"
  #  )
=======

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

>>>>>>> main

    # model eval
    #eval_result = trainer.evaluate(eval_dataset=tokenized_imdb["test"])


    #print(f"Fine tune model accuracy : {eval_result['eval_accuracy']}")

<<<<<<< HEAD
    #return
=======

>>>>>>> main

if __name__ == "__main__":
    main()
