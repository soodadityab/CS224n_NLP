import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging

app = Flask(__name__)
# CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

API_KEY = ""
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

@app.route('/', methods=['GET'])
def home():
    return "Storytelling API Home", 200

@app.route('/generateFinetuned', methods=['POST'])
def generate():
    if not API_KEY:
        return jsonify({"Error": "Please enter an API key"}), 500
    
    data_request = request.get_json()
    prompt = data_request.get("prompt")

    if not prompt:
        return jsonify({"Error": "Please enter a prompt"}), 500

    # payload, defining our model, hyperperams
    payload = {
        "model": "soodadityab@gmail.com/Mistral-7B-v0.1-2024-06-02-23-56-31-20fd5e0e",
        "messages": [{"role": "user", "content": f"{prompt}\nA:"}],
        "temperature": 0.9,
        "max_tokens": 180,
        "n": 2
    }

    # auth
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()

        completions = []
        choices = result.get('choices', [{}])
        for choice in choices:  
            message = choice.get('message', {})
            content = message.get('content', 'N/A')
            completions.append(content)

        return jsonify({"completions": completions})
    else:
        return jsonify({"error": "Did not receive a response from Together API model"}), response.status_code
           
@app.route('/generate', methods=['POST'])
def generatePretrained():
    data_request = request.get_json()
    prompt = data_request.get("prompt")

    if not prompt:
        return jsonify({"Error": "Please enter a prompt"}), 500

    payload = {
        "model": MODEL_NAME,
        "temperature": .9,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        completions = []
        for _ in range(2):
            response = requests.post(TOGETHER_URL, json=payload, headers=headers)
            response_json = response.json()
            completion = response_json['choices'][0]['message']['content']
            completions.append(completion)

        return jsonify({"completions": completions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)