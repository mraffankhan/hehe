import os
import google.generativeai as genai
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
import json # Import json
import re # Import re for parsing

# Import new libraries for file handling
import pypdf
import docx

# Load environment variables (e.g., your API key)
load_dotenv()

app = Flask(__name__)
# --- THIS IS THE UPDATED LINE ---
# Whitelist your Netlify frontend URL
CORS(app, resources={r"/*": {"origins": "https://nlpproject-018.netlify.app"}})
# --- END OF UPDATE ---

# Configure the Gemini API key
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY not found. Please set it in your .env file.")
    exit()

# --- Initialize the generative models ---
# Try 'gemini-1.5-flash' first (modern)
model = None
analysis_model = None

try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    # System instruction for the analysis model (JSON mode)
    ANALYSIS_SYSTEM_INSTRUCTION = """
    You are an expert text analyst. The user will provide a text.
    You must analyze it and return a JSON object with three properties:
    1. "sentiment": (String) The overall sentiment. Must be one of: "Positive", "Negative", "Neutral".
    2. "keywords": (Array of strings) A list of the 5 most important keywords or topics.
    3. "entities": (Array of objects) A list of named entities. Each object must have two properties:
        - "text": (String) The entity text.
        - "type": (String) The entity type (e.g., "Person", "Place", "Organization", "Date", "Product").
    Return ONLY the JSON object.
    """
    # Generation config for forcing JSON output
    JSON_GENERATION_CONFIG = genai.GenerationConfig(
        response_mime_type="application/json",
    )
    analysis_model = genai.GenerativeModel(
        'gemini-1.5-flash',
        system_instruction=ANALYSIS_SYSTEM_INSTRUCTION,
        generation_config=JSON_GENERATION_CONFIG
    )
    print("Successfully initialized gemini-1.5-flash model in JSON mode.")
except Exception as e:
    print(f"Could not initialize 'gemini-1.5-flash' ({e}). Trying 'gemini-1.0-pro'.")
    # Fallback to 'gemini-1.0-pro' (older, broad compatibility)
    try:
        model = genai.GenerativeModel('gemini-1.0-pro')
        analysis_model = genai.GenerativeModel('gemini-1.0-pro')
        print("Successfully initialized gemini-1.0-pro model.")
    except Exception as e2:
        print(f"Could not initialize 'gemini-1.0-pro' ({e2}). Trying 'gemini-pro'.")
        # --- NEW FALLBACK ---
        # Fallback to just 'gemini-pro' (oldest name)
        try:
            model = genai.GenerativeModel('gemini-pro')
            analysis_model = genai.GenerativeModel('gemini-pro')
            print("Successfully initialized gemini-pro model.")
        except Exception as e3:
            print(f"FATAL: Failed to initialize 'gemini-1.5-flash', 'gemini-1.0-pro', AND 'gemini-pro'. {e3}")
            print("Please ensure your API key is correct, your library is updated (`pip install --upgrade google-generativeai`), and your region supports these models.")
            exit()


# --- Helper Function for Error Handling ---
def handle_api_error(e):
    """Handles errors from the Gemini API."""
    print(f"Gemini API Error: {e}")
    # Check for specific blockages (safety, etc.)
    if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
         return jsonify({"error": "Request blocked due to safety settings."}), 400
    if "404" in str(e) or "Model not found" in str(e):
        return jsonify({"error": "Model not found. Your Google AI Studio project might be in a region that doesn't support the required models, or your library is severely outdated."}), 404
    if "429" in str(e): # Handle rate limiting
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    return jsonify({"error": f"An error occurred with the AI model: {e}"}), 500

# --- Helper to extract JSON from model output ---
def extract_json_from_text(text):
    """Extracts a JSON object from a string, even if wrapped in markdown."""
    try:
        # Find the start of the JSON
        json_match = re.search(r'\{', text)
        if not json_match:
            # Try to find JSON array as fallback
            json_match = re.search(r'\[', text)
            if not json_match:
                raise ValueError("No JSON object or array found in the response.")
        
        json_start = json_match.start()
        
        # Find the corresponding end
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

# --- NEW: File Upload Endpoint ---
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
            # Extract text from PDF
            reader = pypdf.PdfReader(file.stream)
            for page in reader.pages:
                extracted_text += page.extract_text() + "\n"
                
        elif file.filename.endswith('.docx'):
            # Extract text from DOCX
            document = docx.Document(file.stream)
            for para in document.paragraphs:
                extracted_text += para.text + "\n"
                
        else:
            return jsonify({"error": "Unsupported file type"}), 400
            
        if not extracted_text.strip():
             return jsonify({"error": "Could not extract text from the file. It might be empty or image-based."}), 400

        return jsonify({"text": extracted_text})

    except Exception as e:
        print(f"File processing error: {e}")
        return jsonify({"error": f"Failed to process file: {e}"}), 500

# --- MODIFIED: Analyze Endpoint ---
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text_to_analyze = data['text']
    
    try:
        # Check if the model is the new one with JSON mode
        if 'gemini-1.5-flash' in analysis_model.model_name:
            print("Using gemini-1.5-flash (JSON mode)")
            response = analysis_model.generate_content(text_to_analyze)
            # response.text is a JSON string, load and re-serialize with jsonify
            json_data = json.loads(response.text)
            return jsonify(json_data)
        else:
            # Build a prompt for gemini-1.0-pro or gemini-pro
            print(f"Using {analysis_model.model_name} (manual JSON mode)")
            prompt = f"""
            You are an expert text analyst. Analyze the following text:
            ---
            {text_to_analyze}
            ---
            Return a single JSON object with three properties:
            1. "sentiment": (String) The overall sentiment. Must be one of: "Positive", "Negative", "Neutral".
            2. "keywords": (Array of strings) A list of the 5 most important keywords or topics.
            3. "entities": (Array of objects) A list of named entities. Each object must have two properties:
                - "text": (String) The entity text.
                - "type": (String) The entity type (e.g., "Person", "Place", "Organization", "Date", "Product").

            Return ONLY the JSON object and nothing else. Do not wrap it in markdown.
            """
            response = analysis_model.generate_content(prompt)
            
            # Manually parse the JSON from the text response
            json_data = extract_json_from_text(response.text)
            return jsonify(json_data)
            
    except Exception as e:
        return handle_api_error(e)


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text_to_summarize = data['text']
    prompt = f"Summarize the following text into a concise paragraph:\n\n{text_to_summarize}"
    
    try:
        response = model.generate_content(prompt)
        return jsonify({"summary": response.text})
    except Exception as e:
        return handle_api_error(e)


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if not data or 'text' not in data or 'question' not in data:
        return jsonify({"error": "Missing text or question"}), 400
    
    text_context = data['text']
    question = data['question']
    prompt = f"Based on the following text, answer the question.\n\nText:\n\"\"\"{text_context}\"\"\"\n\nQuestion: {question}"
    
    try:
        response = model.generate_content(prompt)
        return jsonify({"answer": response.text})
    except Exception as e:
        return handle_api_error(e)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    if not data or 'text' not in data or 'prompt' not in data:
        return jsonify({"error": "Missing text or prompt"}), 400
    
    text_context = data['text']
    user_prompt = data['prompt']
    prompt = f"{user_prompt}\n\nHere is the text to base it on:\n\"\"\"{text_context}\"\"\""
    
    try:
        response = model.generate_content(prompt)
        return jsonify({"generated_text": response.text})
    except Exception as e:
        return handle_api_error(e)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

