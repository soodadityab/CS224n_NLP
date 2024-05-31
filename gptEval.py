# uses gpt4 api to score the quality of stories generated

from openai import OpenAI
import os

client = OpenAI(
    api_key= os.getenv("OPENAI_API_KEY")
)

def query_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()


def main():
    print("Welcome to the GPT interactive query system!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = query_gpt(user_input)
        print(f"GPT: {response}")

if __name__ == "__main__":
    main()