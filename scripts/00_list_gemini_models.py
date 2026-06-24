import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

response = requests.get(url, timeout=60)

if response.status_code != 200:
    raise Exception(f"Gemini API error {response.status_code}: {response.text}")

data = response.json()

print("\nAVAILABLE GEMINI MODELS THAT SUPPORT generateContent:\n")

for model in data.get("models", []):
    name = model.get("name", "")
    methods = model.get("supportedGenerationMethods", [])

    if "generateContent" in methods:
        print(name)