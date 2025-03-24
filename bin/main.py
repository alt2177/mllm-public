from constants import HF_ACCESS_TOKEN, HF_USERNAME

def main():
    # set our collective token
    access_token = HF_ACCESS_TOKEN
    username = HF_USERNAME
    output_repo = "yelp_finetuned_sbatch_upload_test"

    # load and tokenize data
    dataset = load_dataset("yelp_review_full")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", token = access_token)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_yelp = dataset.map(lambda examples:tokenizer(examples["text"], truncation=True,max_length=1024),batched=True)

    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 

    # create model
    model = AutoModelForSequenceClassification.from_pretrained(
        "openai-community/gpt2",
        num_labels=5,
        token = access_token
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # create training-val split
    small_train_dataset = tokenized_yelp["train"].shuffle(seed=42).select(range(10))
    small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42).select(range(10))
 
    output_dir = "yelp_finetune_gpt2_test"
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,

	    hub_model_id=f"{username}/{output_repo}",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        hub_token=access_tokens,
    )

    result = trainer.train()
    print(result)
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
    
	print("================= BEGIN! =================")
	my_mllm = mllm.MLLM()
	my_mllm.train()
	my_mllm.write_results()
	print("================= DONE! =================")


if __name__ == "__main__":
	main()
