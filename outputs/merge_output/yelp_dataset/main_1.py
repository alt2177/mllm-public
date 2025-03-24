from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
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

def main():
    # set our collective token
    access_token = HF_ACCESS_TOKEN 
    username = HF_USERNAME
    output_repo = "yelp_finetuned_gpt2_6gpu_2"

    # load and tokenize data
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)
    
    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    # create model
    model = AutoModelForSequenceClassification.from_pretrained(
        "openai-community/gpt2",
        num_labels=5,
        use_auth_token = access_token
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    #small_train_dataset = tokenized_yelp["train"].shuffle(seed=42).select(range(300000))
    #small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42).select(range(50000))
    tokenized_yelp_train = tokenized_yelp["train"].shuffle(seed=42)
    tokenized_yelp_test = tokenized_yelp["test"].shuffle(seed=42)
    #tokenized_yelp_train = tokenized_yelp["train"].shuffle(seed=42).select(range(1000))
    #tokenized_yelp_test = tokenized_yelp["test"].shuffle(seed=42).select(range(1000))
    HfFolder.save_token(access_token)

    api = HfApi()
    user = api.whoami(token=access_token)
    try:
    	create_repo(f"{username}/{output_repo}", repo_type="model")
    except:
        print('error creating repo for model. it probably already exists')

    output_dir = "yelp_finetune_gpt2_test_gpu6_2"
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=6e-5,    #read that with larger batch size we can increase learning rate
#	gradient_accumulation_steps=4,   #want to explore what this is
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        num_train_epochs=2,
	    fp16=True,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
	hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token,
	push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_yelp_train,
        eval_dataset=tokenized_yelp_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    result = trainer.train()
    print(result)
    predictions = trainer.predict(tokenized_yelp_test)
    logits = predictions.predictions
    probs  = softmax(torch.tensor(logits).float(),dim=1).numpy()
    torch.save(probs,"test_probs_yelp_2.pt")
   # trainer.evaluate()

    HfFolder.save_token(access_token)

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


    #create_repo(f"{username}/{output_repo}", repo_type="model")

    #api.upload_folder(
    #	folder_path=f"./{output_dir}",
#	repo_id=f"{username}/{output_repo}",
 #       repo_type="model"
  #  )

    # model eval
    #eval_result = trainer.evaluate(eval_dataset=tokenized_imdb["test"])


    #print(f"Fine tune model accuracy : {eval_result['eval_accuracy']}")

    #return

if __name__ == "__main__":
    main()
