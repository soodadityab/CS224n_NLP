from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Define the path to the model
model_path = './gpt2-finetuned-writingprompts'

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Function to generate text
def generate_text(prompt, max_length=150, num_beams=5, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=2.0):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example prompts
prompts = [
    "Once upon a time in a land far away,",
    "The quick brown fox jumps over",
    "In the beginning, there was"
]

# Generate and print text for each prompt
for prompt in prompts:
    generated_text = generate_text(prompt)
    print(f"Prompt: {prompt}\nGenerated Text: {generated_text}\n")

