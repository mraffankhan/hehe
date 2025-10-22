import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("🔍 Checking available models...\n")
for m in genai.list_models():
    print("-", m.name)
