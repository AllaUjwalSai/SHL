import os
import requests

# Load environment variables (optional, if using .env files)
# from dotenv import load_dotenv
# load_dotenv()

# Set your Hugging Face API key and model
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

def embed_texts(texts):
    """
    Generate embeddings for a list of texts using Hugging Face API.
    """
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": texts})
    response.raise_for_status()
    return response.json()
