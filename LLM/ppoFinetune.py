# File finetunes the model using PPO with the given user response to the query

import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ppoQueryUser import load_model, main as query_user_main
from createText import generate_text

class TextGenEnv(gym.Env):

    #init the gym
    def __init__(self, model, tokenizer, prompt, user_choice):
        super(TextGenEnv, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.user_choice = user_choice
        self.action_space = spaces.Discrete(2)  # Two choices
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.current_output = None

    #allows ease of resetting env
    def reset(self):
        self.current_output = self.prompt
        return np.array([0])

    #there should be a step function that provides a reward calculation


#updates model based off user selection with PPO
def train_ppo(model_name, prompt, user_choice):
    model, tokenizer = load_model(model_name)
    env = TextGenEnv(model, tokenizer, prompt, user_choice)
    vec_env = DummyVecEnv([lambda: env])

    ppo_model = PPO('MlpPolicy', vec_env, verbose=1)
    ppo_model.learn(total_timesteps=10000)
    ppo_model.save("ppo_textgen")

    return ppo_model

# def fine_tune_model(prompt, ppo_model, model_name):
#     model, tokenizer = load_model(model_name)
#     env = TextGenEnv(model, tokenizer, prompt, None)  # user_choice not needed here
#     obs = env.reset()
#     action, _states = ppo_model.predict(obs)
#     output1 = generate_text(prompt)
#     output2 = generate_text(prompt)
#     outputs = [output1, output2]
#     selected_output = outputs[action]
#     return selected_output

def main():
    model_name, prompt, user_choice = query_user_main()
    ppo_model = train_ppo(model_name, prompt, user_choice)
    print(generate_text(prompt)) #there is no param taking in the new updated PPO model--need to change that

    #fine_tuned_output = fine_tune_model(prompt, ppo_model, model_name)
    #print("Fine-tuned Output:", fine_tuned_output)

if __name__ == "__main__":
    main()
