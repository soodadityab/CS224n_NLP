from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
import signal
import sys
import torch

# use CUDA for GPUs
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
print(f"Using: {device}")

# paths to pre-processed data
train_path = './processed_data/train.wp_combined'
test_path = './processed_data/test.wp_combined'

# keyboard interrupt func
def signal_handler(sig, frame):
    print('KeyboardInterrupt! Exiting gracefully.')
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
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

train_data = load_text_file(train_path)
test_data = load_text_file(test_path)

# load train/test sets post preprocessing
train_dataset = load_dataset('text', data_files={'train': train_path})['train']
test_dataset = load_dataset('text', data_files={'test': test_path})['test']

# tokenize datasets
def tokenize_function(dataset):
    return tokenizer(dataset['text'], truncation=True, padding='max_length', max_length=128)

try:
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
    test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=4)
except KeyboardInterrupt:
    print('KeyboardInterrupt! Exiting gracefully.')
    sys.exit(0)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    )

# training args
training_args = TrainingArguments(
    output_dir='./results/GPT2',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    logging_dir='./logs/GPT2',
    fp16=True,
    gradient_accumulation_steps=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss' 
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

try:
    # train model
    trainer.train()

    # save to disk
    model.save_pretrained("./gpt2-finetuned-writingprompts")
    tokenizer.save_pretrained("./gpt2-finetuned-writingprompts")
except KeyboardInterrupt:
    print('KeyboardInterrupt (training)! Exiting gracefully.')
    sys.exit(0)

print("Fine-tuning complete and saved.")