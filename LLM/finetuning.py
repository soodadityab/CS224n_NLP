
# IMPORANT: THIS SCRIPT NEEDS TO SAVE TOKENIZER. CANNOT USE DEFAULT AS NEW TKOENS ARE INTRODUCED.

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
import signal
import sys
import torch

# use CUDA library for GPU training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# paths to pre-processed data
train_path = './processed_data/train.wp_combined'
test_path = './processed_data/test.wp_combined'

# keyboard interrupt
def signal_handler(sig, frame):
    print('KeyboardInterrupt caught! Exiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# get tokenizer. this is from HuggingFace
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


# specify padding--ensures all sentences in a batch are of same length
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# get model from HuggingFace
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model.to(device) 

# further preprocessing--removes whitespace
def load_text_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

train_data = load_text_file(train_path)
test_data = load_text_file(test_path)

train_dataset = load_dataset('text', data_files={'train': train_path})['train']
test_dataset = load_dataset('text', data_files={'test': test_path})['test']

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

try:
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)  # Increase num_proc for parallel processing
    test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=4)   # Increase num_proc for parallel processing
except KeyboardInterrupt:
    print('KeyboardInterrupt caught during dataset mapping! Exiting gracefully...')
    sys.exit(0)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Increase batch size if possible
    per_device_eval_batch_size=8,   # Increase batch size if possible
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    logging_dir='./logs',
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=2,  # Use gradient accumulation if batch size cannot be increased
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

try:
    # Train the model
    trainer.train()

    model.save_pretrained("./gpt2-finetuned-writingprompts")
    tokenizer.save_pretrained("./gpt2-finetuned-writingprompts")
except KeyboardInterrupt:
    print('KeyboardInterrupt caught during training! Exiting gracefully...')
    sys.exit(0)

