# File queries user choice between two generated outputs for the existing model
from transformers import AutoModelForCausalLM, AutoTokenizer
from createText import generate_text

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# def generate_outputs(prompt, model, tokenizer):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs1 = model.generate_text(**inputs, max_length=50, num_return_sequences=1)
#     outputs2 = model.generate_text(**inputs, max_length=50, num_return_sequences=1)
    
#     output_text1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
#     output_text2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
    
#    return output_text1, output_text2

def prompt_user(output1, output2):
    print("Output 1:", output1)
    print("Output 2:", output2)
    choice = input("Choose the better output (1 or 2): ")
    return int(choice) - 1

def main():
    model_name = './gpt2-finetuned-writingprompts'
    prompt = "Once upon a time"
    
    model, tokenizer = load_model(model_name)
    output1 = generate_text(prompt)
    output2 = generate_text(prompt)
    user_choice = prompt_user(output1, output2)
    
    return model_name, prompt, user_choice

if __name__ == "__main__":
    main()
