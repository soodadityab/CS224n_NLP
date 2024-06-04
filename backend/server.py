import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging

app = Flask(__name__)
CORS(app)

API_KEY = "35ba5bebf6288e43fdc8989965161592e3335d7067c772c0c6995cdc0e60cd88"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

@app.route('/', methods=['GET'])
def home():
    return "Storytelling API Home", 200

@app.route('/generate', methods=['POST'])
def generate():
    if not API_KEY:
        return jsonify({"error": "API key not found"}), 500
    
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "Prompt not provided"}), 400

    # payload
    payload = {
        "model": "soodadityab@gmail.com/Mistral-7B-v0.1-2024-06-02-23-56-31-20fd5e0e",
        "messages": [{"role": "user", "content": f"Q: {prompt}\nA:"}],
        "temperature": 0.8,
        "max_tokens": 120,
        "n": 2  # Requesting two completions
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        completions = [choice.get('message', {}).get('content', 'No response') for choice in result.get('choices', [{}])]
        return jsonify({"completions": completions})
    else:
        return jsonify({"error": "Failed to get response from Together API"}), response.status_code


# @app.route('/retrain_model', methods=['POST'])
# def retrain_model():

           
# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.get_json()
#     prompt = data.get("prompt")

#     if not prompt:
#         return jsonify({"error": "No prompt provided"}), 400

#     payload = {
#         "model": MODEL_NAME,
#         "temperature": 1.0,
#         "frequency_penalty": 0,
#         "presence_penalty": 0,
#         "messages": [{"role": "user", "content": prompt}]
#     }
#     headers = {
#         "accept": "application/json",
#         "content-type": "application/json",
#         "Authorization": f"Bearer {API_KEY}"
#     }

#     try:
#         completions = []
#         for _ in range(2):
#             response = requests.post(TOGETHER_URL, json=payload, headers=headers)
#             response_json = response.json()
#             completion = response_json['choices'][0]['message']['content']
#             completions.append(completion)

#         return jsonify({"completions": completions})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/feedback', methods=['POST'])
# def feedback():
#     data = request.get_json()
#     feedback = data.get("feedback")

#     if feedback is None:
#         return jsonify({"error": "No feedback provided"}), 400

#     print(f"Feedback received: {feedback}")

#     return jsonify({"message": "Feedback received"}), 200

if __name__ == '__main__':
    app.run(debug=True)
