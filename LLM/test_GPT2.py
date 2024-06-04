from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Define paths to preprocessed data
test_path = './processed_data/test.wp_combined'
model_path = './gpt2-finetuned-writingprompts'

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_path)
model.to(device)

# Load the test dataset
test_dataset = load_dataset('text', data_files={'test': test_path})['test']

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

try:
    test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=4)
except ValueError as e:
    print(e)

test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments (for evaluation, some parameters are not needed)
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,  # Adjust based on your GPU memory
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=test_dataset
)

# Evaluate the model
eval_results = trainer.evaluate()

print("Evaluation Results:", eval_results)

