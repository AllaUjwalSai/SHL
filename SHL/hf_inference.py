import os
import requests

# Environment variables for your Hugging Face API token and model name
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "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")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")




def run_llm(prompt):
    """
    Call the Hugging Face Inference API to generate text based on the prompt.
    """
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt}
    api_url = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error: {response.status_code} {response.text}"

    data = response.json()
    generated_text = ""
    if isinstance(data, list) and "generated_text" in data[0]:
        generated_text = data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        generated_text = data["generated_text"]
    return generated_text
