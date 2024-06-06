from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# FUNCTIONS
def create_text(prompt, model, max_length=200, num_beams=5, temperature=0.9, top_k=40, top_p=.9, repetition_penalty=1.5):
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

# use CUDA for GPUs
device = "cpu"
if torch.cuda.is_available():
  device = "cuda"
print(f"Using: {device}")

model_path = './gpt2-finetuned-writingprompts'
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer = GPT2Tokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()

prompts = [
    "Once upon a time,",
    "During the horror movie I was watching with my family,",
    "At first, it was the loud noises and then"
]

for prompt in prompts:
    created_text = create_text(prompt)
    print(f"Prompt: {prompt}\nGenerated Text: {created_text}\n")

