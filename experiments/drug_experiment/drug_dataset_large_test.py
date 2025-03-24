from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset,Dataset
import evaluate
import numpy as np
import torch
from torch.nn.functional import softmax
import os
import shutil

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def main():
    # set our collective token and HF info
    access_token = HF_ACCESS_TOKEN 
    username = HF_USERNAME

    # load and tokenize data
    dataset = load_dataset("lewtun/drug-reviews")
    train_data = dataset["train"]
    eval_data = dataset["test"]
    reviews_train = train_data['review']
    ratings_train = train_data['rating']
    reviews_eval = eval_data['review']
    ratings_eval = eval_data['rating']
    drug_data_train = {'label': [int(rating-1) for rating in ratings_train],'text': reviews_train}
    drug_data_train = Dataset.from_dict(drug_data_train)
    drug_data_eval = {'label': [int(rating-1) for rating in ratings_eval],'text': reviews_eval}
    drug_data_eval = Dataset.from_dict(drug_data_eval)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl", token = access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_drug_train = drug_data_train.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024),batched=True)
    tokenized_drug_test = drug_data_eval.map(lambda examples:tokenizer(examples["text"], truncation=True, max_length=1024),batched=True)
    # pad tokens
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) 
    # Start index for the first subset     
    # create model
    model = AutoModelForSequenceClassification.from_pretrained(
        "openai-community/gpt2-xl",
        num_labels=10,
        token = access_token
    )
    model.config.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    # repo that will be made on huggingface
    output_repo = f"gpt2_f_experiment_drug_data_large_test"

    # Select 5% of the train data for finetuning a specific model
    
    train_dataset = tokenized_drug_train.shuffle(seed=42)
    eval_dataset = tokenized_drug_test.shuffle(seed=42)
    
    
    # Define the size of the validation set (e.g., 20% of the total data)
    validation_size = int(0.2 * len(eval_dataset))
    indices = list(range(len(eval_dataset)))
    test_indices = indices[validation_size:]
    validation_indices = indices[:validation_size]
    
    # Split the shuffled training dataset into training and validation sets
    test_dataset = eval_dataset.select(test_indices)
    validation_dataset = eval_dataset.select(validation_indices)
    
    # Get all indices of the original dataset
    #all_indices = list(range(len(tokenized_yelp["train"])))
    #indices_not_used = list(set(all_indices) - set(list(range(start_index, end_index))))
    #train_validation_dataset = tokenized_yelp["train"].select(indices_not_used)
    print(len(train_dataset))
    print(len(validation_dataset))
    print(len(test_dataset))
    # create the repo before we try to push the model to huggingface
    HfFolder.save_token(access_token)
    api = HfApi()
    user = api.whoami(token=access_token)
    try:
        create_repo(f"{username}/{output_repo}", repo_type="model")
    except:
        print('error creating repo for model. it probably already exists')

    output_dir = "tam_test_out_drug_data_large_test"
    # training loop
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=6e-5,    #read that with larger batch size we can increase learning rate
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        fp16=True,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        #evaluation_strategy="no",
        #do_eval=False,
        save_strategy="epoch",
        load_best_model_at_end=True,
        hub_model_id=f"{username}/{output_repo}",
        hub_token = access_token,
        push_to_hub=False
    )
    # Note that currently we use the validation_data as the eval set
    # It doesn't really matter since we get the test results later
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print('************TRAINING STARTED*************')
    result = trainer.train()
    print('*************TRAINING COMPLETED**************')
    
    # Check the peak GPU memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    print("Peak GPU memory usage:", peak_memory / 1024**2, "MB") 
    
    # Save the output probabilities for merging outputs
    # use ALL of the training samples
    
    # UNCOMMENT FOR FINAL RUN
    # predictions = trainer.predict(tokenized_drug_train)
    # logits = predictions.predictions
    # probs = softmax(torch.tensor(logits).float(),dim=1).numpy()
    # torch.save(probs,f"final_layer_predictions/{output_repo}_train.pt")
    # print(predictions)

    # predictions = trainer.predict(tokenized_drug_test)
    # logits = predictions.predictions
    # probs = softmax(torch.tensor(logits).float(),dim=1).numpy()
    # torch.save(probs,f"final_layer_predictions/{output_repo}_test.pt")

    validation_result = trainer.evaluate(eval_dataset = validation_dataset)
    test_result = trainer.evaluate(eval_dataset = test_dataset)

    # Save the accuracy of each model for later comparison
    f = open("accuracy_large.txt", "a")
    f.write(f"Large Fine tune model  : {result}\n")
    #f.write(f"Large Fine tune model {m} all train data result : {predictions.metrics}\n")
    f.write(f"Large Fine tine model  validation result : {validation_result}\n")  
    f.write(f"Large Fine tune model  test results : {test_result}\n\n")
    f.close()

    # When push to hub is false, the model is saved under a folder that
    # starts with "checkpoint"
    directories = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    checkpoint_directories = [d for d in directories if d.startswith("checkpoint")]

    api.upload_folder(
        folder_path=os.path.join(output_dir, checkpoint_directories[0]),
        repo_id=f"{username}/{output_repo}",
        repo_type="model",
    )
    
    # # Specify the path of the folder you want to remove
    # folder_path = os.path.join(output_dir, checkpoint_directories[0])

    # # Check if the folder exists before attempting to remove it
    # if os.path.exists(folder_path):
    #     # Remove the folder
    #     shutil.rmtree(folder_path)
    #     print("Folder removed successfully.")
    # else:
    #     print("Folder does not exist.")

if __name__ == "__main__":
    main()
