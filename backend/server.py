from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/username/interactive-storytelling-model"
HUGGING_FACE_API_KEY = "your_hugging_face_api_key"

headers = {
    "Authorization": f"Bearer {HUGGING_FACE_API_KEY}"
}

@app.route('/generate', methods=['POST'])
def generate_story():
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    response = requests.post(HUGGING_FACE_API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code != 200:
        return jsonify({"error": "Error generating story"}), response.status_code
    story = response.json()[0]['generated_text']
    return jsonify({"story": story})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    feedback = data.get('feedback')
    if feedback is None:
        return jsonify({"error": "Feedback is required"}), 400
    # Process feedback (e.g., store it in a database)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
