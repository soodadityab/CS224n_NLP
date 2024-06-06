# uses gpt4 api to score the quality of stories generated
#scores on five metrics, runs 100 times, outputs mean SD for each model
from openai import OpenAI
import os
import re

client = OpenAI(

)

def query(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def score_story(story):
    prompt = f"Please rate the following story on a scale from 1 to 10 based on its coherence, creativity, human-like narration, and engagement:\n\n{story}. Please rate with 0.2 increments with an interesting story centered around a score of 5.0. \n\nYour rating:"
    response = query(prompt)
    print(f"raw GPT response: {response}")
    # regex pattern matching
    match = re.search(r'\b(\d+\.\d+|\d+)\b', response)
    if match:
        score = float(match.group(1))
        if 1 <= score <= 10:
            return score
        else:
            print("Score is out of range:", score)
            return None
    else:
        print("Invalid score received:", response)
        return None
    # try:
    #     score = int(response)
    #     if 1 <= score <= 10:
    #         return score
    #     else:
    #         raise ValueError("Score out of range")
    # except ValueError:
    #     print("Invalid score received:", response)
    #     return None
    

def main():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = query(user_input)
        print(f"GPT: {response}")

        example_story = response
        score = score_story(example_story)
        if score is not None:
            print(f"Score: {score}")

if __name__ == "__main__":
    main()