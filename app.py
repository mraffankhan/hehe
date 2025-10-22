import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import json
import re
import pypdf
import docx

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------------------------------------------------
# ✅ CONFIGURE GEMINI API
# ---------------------------------------------------------------------
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("✅ GEMINI_API_KEY loaded successfully.")
except KeyError:
    print("❌ GEMINI_API_KEY not found in .env file.")
    exit()

# ---------------------------------------------------------------------
# ✅ AUTO-DETECT SUPPORTED MODEL
# ---------------------------------------------------------------------
try:
    available_models = [m.name for m in genai.list_models()]
    print("\nAvailable models for this key:")
    for m in available_models:
        print("-", m)

    # Prefer the newest 2.5 models
    if 'models/gemini-2.5-flash-lite-preview-09-2025' in available_models:
        selected_model = 'models/gemini-2.5-flash-lite-preview-09-2025'
    elif 'models/gemini-2.5-flash' in available_models:
        selected_model = 'models/gemini-2.5-flash'
    elif 'models/gemini-2.5-flash-live-preview' in available_models:
        selected_model = 'models/gemini-2.5-flash-live-preview'
    elif 'models/gemini-2.5-flash-native-audio-latest' in available_models:
        selected_model = 'models/gemini-2.5-flash-native-audio-latest'
    else:
        raise Exception("❌ No supported Gemini 2.5 text model found for your region/API key.")

    print(f"\n✅ Using model: {selected_model}")

    model = genai.GenerativeModel(selected_model)
    analysis_model = genai.GenerativeModel(selected_model)
except Exception as e:
    print(f"FATAL: Could not initialize Gemini model: {e}")
    exit()

# ---------------------------------------------------------------------
# ✅ HELPER FUNCTIONS
# ---------------------------------------------------------------------
def handle_api_error(e):
    """Handle Gemini API errors gracefully."""
    print(f"Gemini API Error: {e}")
    if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
        return jsonify({"error": "Request blocked due to safety settings."}), 400
    if "404" in str(e) or "Model not found" in str(e):
        return jsonify({"error": "Model not found. Your region may not support this model."}), 404
    if "429" in str(e):
        return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
    return jsonify({"error": f"An error occurred: {e}"}), 500


def extract_json_from_text(text):
    """Extract JSON object from text output."""
    try:
        json_match = re.search(r'\{', text)
        if not json_match:
            json_match = re.search(r'\[', text)
            if not json_match:
                raise ValueError("No JSON object or array found.")
        json_start = json_match.start()

        if text[json_start] == '{':
            end_char = '}'
            last_match = re.search(r'\}', text[::-1])
        else:
            end_char = ']'
            last_match = re.search(r'\]', text[::-1])

        if not last_match:
            raise ValueError(f"No closing '{end_char}' found for JSON.")
        json_end = len(text) - last_match.start()

        json_string = text[json_start:json_end]
        return json.loads(json_string)
    except Exception as e:
        print(f"JSON Parsing Error: {e}\nRaw text: {text}")
        raise ValueError("Failed to parse JSON from model response.")

# ---------------------------------------------------------------------
# ✅ FILE UPLOAD → EXTRACT TEXT
# ---------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    extracted_text = ""
    try:
        if file.filename.endswith('.pdf'):
            reader = pypdf.PdfReader(file.stream)
            for page in reader.pages:
                extracted_text += page.extract_text() + "\n"
        elif file.filename.endswith('.docx'):
            document = docx.Document(file.stream)
            for para in document.paragraphs:
                extracted_text += para.text + "\n"
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        if not extracted_text.strip():
            return jsonify({"error": "Could not extract text from file."}), 400

        return jsonify({"text": extracted_text})
    except Exception as e:
        print(f"File processing error: {e}")
        return jsonify({"error": f"Failed to process file: {e}"}), 500

# ---------------------------------------------------------------------
# ✅ TEXT ANALYSIS ENDPOINT
# ---------------------------------------------------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text_to_analyze = data['text']
    prompt = f"""
    You are an expert text analyst. Analyze the following text:
    ---
    {text_to_analyze}
    ---
    Return ONLY a JSON object with:
    1. "sentiment": "Positive", "Negative", or "Neutral"
    2. "keywords": [Top 5 important keywords]
    3. "entities": [{{"text": "Entity", "type": "Type"}}]
    """

    try:
        response = analysis_model.generate_content(prompt)
        return jsonify(extract_json_from_text(response.text))
    except Exception as e:
        return handle_api_error(e)

# ---------------------------------------------------------------------
# ✅ SUMMARIZATION ENDPOINT
# ---------------------------------------------------------------------
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text_to_summarize = data['text']
    prompt = f"Summarize this text in one concise paragraph:\n\n{text_to_summarize}"
    try:
        response = model.generate_content(prompt)
        return jsonify({"summary": response.text})
    except Exception as e:
        return handle_api_error(e)

# ---------------------------------------------------------------------
# ✅ Q&A ENDPOINT
# ---------------------------------------------------------------------
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if not data or 'text' not in data or 'question' not in data:
        return jsonify({"error": "Missing text or question"}), 400

    text_context = data['text']
    question = data['question']
    prompt = f"Based on this text:\n\"\"\"{text_context}\"\"\"\n\nAnswer: {question}"
    try:
        response = model.generate_content(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return handle_api_error(e)

# ---------------------------------------------------------------------
# ✅ TEXT GENERATION ENDPOINT
# ---------------------------------------------------------------------
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    if not data or 'text' not in data or 'prompt' not in data:
        return jsonify({"error": "Missing text or prompt"}), 400

    text_context = data['text']
    user_prompt = data['prompt']
    prompt = f"{user_prompt}\n\nContext:\n\"\"\"{text_context}\"\"\""
    try:
        response = model.generate_content(prompt)
        return jsonify({"generated_text": response.text})
    except Exception as e:
        return handle_api_error(e)

# ---------------------------------------------------------------------
# ✅ RUN FLASK SERVER
# ---------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
