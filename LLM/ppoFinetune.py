# File finetunes the model using PPO with the given user response to the query
# fine_tune.py
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ppoQueryUser import load_model, generate_outputs, main as query_user_main

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
        outputs = generate_outputs(self.prompt, self.model, self.tokenizer)
        selected_output = outputs[action]
        
        reward = 1 if action == self.user_choice else 0  # Simple reward based on user choice
        done = True
        return np.array([reward]), reward, done, {}

def train_ppo(model_name, prompt, user_choice):
    model, tokenizer = load_model(model_name)
    env = TextGenEnv(model, tokenizer, prompt, user_choice)
    vec_env = DummyVecEnv([lambda: env])

    ppo_model = PPO('MlpPolicy', vec_env, verbose=1)
    ppo_model.learn(total_timesteps=10000)
    ppo_model.save("ppo_textgen")

    return ppo_model

def fine_tune_model(prompt, ppo_model, model_name):
    model, tokenizer = load_model(model_name)
    env = TextGenEnv(model, tokenizer, prompt, None)  # user_choice not needed here
    obs = env.reset()
    action, _states = ppo_model.predict(obs)
    outputs = generate_outputs(prompt, model, tokenizer)
    selected_output = outputs[action]
    return selected_output

def main():
    model_name, prompt, user_choice = query_user_main()
    ppo_model = train_ppo(model_name, prompt, user_choice)
    fine_tuned_output = fine_tune_model(prompt, ppo_model, model_name)
    print("Fine-tuned Output:", fine_tuned_output)

if __name__ == "__main__":
    main()
