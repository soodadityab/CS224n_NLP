from together import Together
import os

# uses the env variable
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

# start the fine-tuning job
resp = client.fine_tuning.create(
  training_file='file-ec995bba-8af8-4dab-9a6a-81f78c27153e',
  model='meta-llama/Meta-Llama-3-8B',
  n_epochs=3,
  n_checkpoints=1,
  batch_size=4,
  learning_rate=3e-5
)

fine_tune_id = resp.id
print(f"Fine-tuning job ID: {fine_tune_id}")

