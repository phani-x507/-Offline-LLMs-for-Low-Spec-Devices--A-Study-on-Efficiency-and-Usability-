import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import numpy as np
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models_offline"

clm_models = {
    "TinyLLaMA": f"{MODEL_DIR}/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2": f"{MODEL_DIR}/microsoft-phi-2",
    "Gemma-2B": f"{MODEL_DIR}/google-gemma-2b"
}

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
sample_texts = dataset["text"][:5]

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def calculate_perplexity(log_probs):
    return math.exp(-np.mean(log_probs))

def evaluate_clm(model_name, sample_texts):
    tokenizer = AutoTokenizer.from_pretrained(clm_models[model_name], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(clm_models[model_name], local_files_only=True).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions = []
    references = []
    log_likelihoods = []

    for text in sample_texts:
        if not text.strip():
            continue

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
        if inputs["input_ids"].shape[1] == 0:
            continue
        
        outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        with torch.no_grad():
            logits = model(inputs["input_ids"]).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
            log_likelihood = token_log_probs.mean().item()
            log_likelihoods.append(log_likelihood)

        predictions.append(generated_text)
        references.append([text])

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"] if predictions else 0
    rouge_scores = rouge_metric.compute(predictions=predictions, references=[ref[0] for ref in references]) if predictions else {}
    perplexity = calculate_perplexity(log_likelihoods) if log_likelihoods else float('inf')

    return {
        "Model": model_name,
        "BLEU Score": bleu_score,
        "ROUGE-1": rouge_scores.get("rouge1", 0),
        "ROUGE-2": rouge_scores.get("rouge2", 0),
        "ROUGE-L": rouge_scores.get("rougeL", 0),
        "Perplexity": perplexity
    }

def evaluate_all_clms():
    return {name: evaluate_clm(name, sample_texts) for name in clm_models}
