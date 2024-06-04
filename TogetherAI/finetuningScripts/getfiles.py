import os
from together import Together

togetherAI = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

print(togetherAI.files.list())
