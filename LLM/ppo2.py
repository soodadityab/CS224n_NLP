import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import AutoModelForCausalLM, AutoTokenizer

class TextGenEnv(gym.Env):
    def __init__(self, model, tokenizer, prompt, user_choice):
        super(TextGenEnv, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.user_choice = user_choice
        self.action_space = spaces.Discrete(2)  # Two choices
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.current_output = None

    def reset(self):
        self.current_output = self.prompt
        return np.array([0])

    def step(self, action):
        output1 = generate_text(self.prompt, self.model, self.tokenizer)
        output2 = generate_text(self.prompt, self.model, self.tokenizer)
        outputs = [output1, output2]

        self.current_output = outputs[action]

        reward = 1 if action == self.user_choice else 0
        done = True
        return np.array([1]), reward, done, {}

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def train_ppo(model_name, prompt, user_choice):
    model, tokenizer = load_model(model_name)
    env = TextGenEnv(model, tokenizer, prompt, user_choice)
    vec_env = DummyVecEnv([lambda: env])

    ppo_model = PPO('MlpPolicy', vec_env, verbose=1)
    ppo_model.learn(total_timesteps=8000)
    ppo_model.save("ppo_textgen")

    model.save_pretrained("./llama3-ppo-finetuned")
    tokenizer.save_pretrained("./llama3-ppo-finetuned")

    return ppo_model

def query_user_main():
    model_name = 'meta-llama/Meta-Llama-3-8B'  # Using the provided model name from Hugging Face
    prompt = "Once upon a time"

    model, tokenizer = load_model(model_name)
    output1 = generate_text(prompt, model, tokenizer)
    output2 = generate_text(prompt, model, tokenizer)
    user_choice = prompt_user(output1, output2)

    return model_name, prompt, user_choice

def generate_text(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def prompt_user(output1, output2):
    print("Output 1:", output1)
    print("Output 2:", output2)
    choice = input("Choose the better output (1 or 2): ")
    return int(choice) - 1

def main():
    model_name, prompt, user_choice = query_user_main()
    ppo_model = train_ppo(model_name, prompt, user_choice)
    print(generate_text(prompt, ppo_model.actor_critic, ppo_model.actor_critic.tokenizer))

if __name__ == "__main__":
    main()
