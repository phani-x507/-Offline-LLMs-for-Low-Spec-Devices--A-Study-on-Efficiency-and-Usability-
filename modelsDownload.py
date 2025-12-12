from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import os

# Define models to download
models = {
    "DistilBERT": "distilbert-base-uncased",
    "MobileBERT": "google/mobilebert-uncased",
    "Gemma 2B": "google/gemma-2b",
    "Phi-2": "microsoft/phi-2",
    "TinyLLaMA": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

# Cache folder
cache_dir = "./cache_models"
os.makedirs(cache_dir, exist_ok=True)

# Download models & tokenizers
for name, model_id in models.items():
    print(f"Downloading {name} ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if "bert" in model_id or "TinyGPT" in model_id:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    tokenizer.save_pretrained(f"{cache_dir}/{name}")
    model.save_pretrained(f"{cache_dir}/{name}")

print("✅ All models saved locally!")
