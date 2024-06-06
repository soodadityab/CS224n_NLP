from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

def tokenize_function(dataset):
    return tokenizer(dataset['text'], truncation=True, padding='max_length', max_length=128)

# use CUDA for GPUs
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
print(f"Using: {device}")

test_path = './processed_data/test.wp_combined'
model_path = './gpt2-finetuned-writingprompts'

# get finetuned model + tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)

testing_data = load_dataset('text', data_files={'test': test_path})['test']

try:
    testing_data = testing_data.map(tokenize_function, batched=True, num_proc=4)
except ValueError as e:
    print(e)

testing_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=testing_data
)

eval_results = trainer.evaluate()

print("Evaluation Results:", eval_results)

