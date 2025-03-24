
from huggingface_hub import notebook_login
notebook_login()

# |%%--%%| <aEBRj5y7cp|E6oW6sorFO>

from datasets import load_dataset
dataset = load_dataset("yelp_review_full")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# |%%--%%| <E6oW6sorFO|118IHHKwsg>

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True,max_length=1024)

# |%%--%%| <118IHHKwsg|S5zq9gqocM>

tokenized_yelp = dataset.map(preprocess_function, batched=True)

# |%%--%%| <S5zq9gqocM|q1ytGGkN5p>

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# |%%--%%| <q1ytGGkN5p|ETRU2ImSgM>

import evaluate
accuracy = evaluate.load("accuracy")

# |%%--%%| <ETRU2ImSgM|1mEuTDI46L>

import numpy as np
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# |%%--%%| <1mEuTDI46L|OSVKEq1N8j>

# id2label = {0: "NEGATIVE", 1: "POSITIVE"}
# label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# |%%--%%| <OSVKEq1N8j|cUIeFI3wiy>

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    "openai-community/gpt2", num_labels=5
)
model.config.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

# |%%--%%| <cUIeFI3wiy|hllTKn7jAt>

small_train_dataset = tokenized_yelp["train"].shuffle(seed=42).select(range(300000))
small_eval_dataset = tokenized_yelp["test"].shuffle(seed=42).select(range(50000))

# |%%--%%| <hllTKn7jAt|qyczHPanlV>

training_args = TrainingArguments(
    output_dir="yelp_finetune_epoch_1_gpt2_1",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# |%%--%%| <qyczHPanlV|E1omsgmmDg>

eval_result = trainer.evaluate(eval_dataset=tokenized_imdb["test"])

# |%%--%%| <E1omsgmmDg|BQDDXeGDyu>

print(f"Fine tune model accuracy : {eval_result['eval_accuracy']}")

# |%%--%%| <BQDDXeGDyu|V9WtsSbZPg>


