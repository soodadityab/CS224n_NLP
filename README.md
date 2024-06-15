# CS224n_NLP
Final CS224n project: using reinforcement learning (PPO) and transformers (GPT) finetuning for interactive story telling.

Due to nascent developments in Natural Language Processing (NLP), transformer-based large language models (LLMs) have become increasingly adept at generating human-like text to perform tasks like neural machine translation. However, these models fall short in their ability to generate cohesive and engaging stories. Creative storytelling demands not only syntactic precision but also a subtle understanding of narrative structures, character development, and original creativity—elements that current models struggle to emulate.

Enhancing the story-telling capabilities of language models serves practical purposes for various use cases such as entertainment, education, and therapeutic applications. Stories have historically been essential to human culture, serving as a medium of communication and a powerful tool for conveying complex ideas and emotions.

In this study, we focused on four prominent SOTA language models: GPT-2, Meta Llama 3 8B, Mistral 7B, and Solar 10.7B. Each of these models embodies a different architecture and approach to language modeling, providing a diverse set of capabilities and limitations. Our objective is to evaluate and enhance these models' story-telling performance through a systematic approach involving fine-tuning and Proximal Policy Optimization (PPO) reinforcement learning (RL).

We utilized a robust database of human-written stories and prompts to fine-tune each model. This process aims to imbue the models with a deeper understanding of narrative elements and improve their ability to generate coherent and engaging texts. Using a web interface to implement PPO RL, we tailored the model to human feedback for adaptive story-telling. This interface acts as a mechanism to collect data for reinforcement learning by tokenizing user feedback into a reward signal for the LLM. In doing so, we are able to incorporate human preferences into the training process, aligning the models' outputs more closely with human expectations and improving their overall narrative quality.

required installations:
- numpy
- gym
- stable_baselines3
- openai
