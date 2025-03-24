from huggingface_hub import HfApi, HfFolder, create_repo
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
import numpy as np
import torch
import os
import shutil
from constants import HF_ACCESS_TOKEN, HF_USERNAME, 


rouge = evaluate.load("rouge")
prefix = "summarize: "

def preprocess_function(examples,tokenizer):
    inputs = [prefix + doc for doc in examples["text"]]
    #inputs = [doc + "\nTL;DR:\n" for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred,tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

def main():
    # set our collective token and HF info
    access_token = HF_ACCESS_TOKEN 
    username = HF_USERNAME

    # load and tokenize data
    billsum = load_dataset("billsum", split="ca_test")
    billsum = billsum.train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small",token=access_token)
    tokenized_billsum = billsum.map(lambda examples:preprocess_function(examples,tokenizer),batched=True)    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer) 
    # Start index for the first subset
    start_index = 0
    for m in range(5):
        
        torch.cuda.empty_cache()
        # Check the amount of GPU memory currently allocated
        allocated_memory = torch.cuda.memory_allocated()
        print("Allocated GPU memory:", allocated_memory / 1024**2, "MB")

        # Check the peak GPU memory usage
        peak_memory = torch.cuda.max_memory_allocated()
        print("Peak GPU memory usage:", peak_memory / 1024**2, "MB")
        # create model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google-t5/t5-small",
            token = access_token
        )

        # repo that will be made on huggingface
        output_repo = f"t5_f_experiment_{m}"

        # Select 5% of the train data for finetuning a specific model
        num_of_samples = int(len(tokenized_billsum["train"]) * 0.2)
        end_index = start_index + num_of_samples
        
        train_dataset = tokenized_billsum["train"].shuffle(seed=42).select(range(start_index, end_index))
        eval_dataset = tokenized_billsum["test"].shuffle(seed=42)
        
        start_index = end_index
        
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

        output_dir = "."
        # training loop
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,    #read that with larger batch size we can increase learning rate
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            fp16=True,
            weight_decay=0.01,
            predict_with_generate=True,
            evaluation_strategy="epoch",
            #evaluation_strategy="no",
            #do_eval=False,
            save_strategy="epoch",
            load_best_model_at_end=True,
            hub_model_id=f"{username}/{output_repo}",
            hub_token = access_token,
            push_to_hub=False,
            report_to="none",
        )
        # Note that currently we use the validation_data as the eval set
        # It doesn't really matter since we get the test results later
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda x : compute_metrics(x,tokenizer),
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
        # predictions = trainer.predict(tokenized_yelp["train"])
        # logits = predictions.predictions
        # probs = softmax(torch.tensor(logits).float(),dim=1).numpy()
        # torch.save(probs,f"final_layer_predictions/{output_repo}_train.pt")
        # print(predictions)

        # predictions = trainer.predict(tokenized_yelp["test"])
        # logits = predictions.predictions
        # probs = softmax(torch.tensor(logits).float(),dim=1).numpy()
        # torch.save(probs,f"final_layer_predictions/{output_repo}_test.pt")

        validation_result = trainer.evaluate(eval_dataset = validation_dataset)
        test_result = trainer.evaluate(eval_dataset = test_dataset)

        # Save the accuracy of each model for later comparison
        f = open("accuracy.txt", "a")
        f.write(f"Fine tune model {m} : {result}\n")
        f.write(f"Fine tine model {m} validation result : {validation_result}\n")  
        f.write(f"Fine tune model {m} test results : {test_result}\n\n")
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
        
        # Specify the path of the folder you want to remove
        folder_path = os.path.join(output_dir, checkpoint_directories[0])

        # Check if the folder exists before attempting to remove it
        if os.path.exists(folder_path):
            # Remove the folder
            shutil.rmtree(folder_path)
            print("Folder removed successfully.")
        else:
            print("Folder does not exist.")

if __name__ == "__main__":
    main()
