import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


dataset = load_dataset("imdb")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


training_args = TrainingArguments(
    output_dir="./results",          
    evaluation_strategy="epoch",     
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=8,    
    num_train_epochs=3,              
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)


trainer.train()

trainer.evaluate()


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions

# Example of prediction
text = "The movie was amazing and I loved it!"
pred = predict(text)
print("Prediction:", pred)



