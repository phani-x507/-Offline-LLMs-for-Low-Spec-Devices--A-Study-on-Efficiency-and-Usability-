from flask import Flask, render_template
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import numpy as np
import math

app = Flask(__name__)

# Define local model directories
MODEL_DIR = "models_offline"

device = "cuda" if torch.cuda.is_available() else "cpu"

# MLM Models
mlm_models = {
    "DistilBERT": f"{MODEL_DIR}/distilbert-base-uncased",
    "MobileBERT": f"{MODEL_DIR}/google-mobilebert-uncased"
}

# CLM Models
clm_models = {
    "TinyLLaMA": f"{MODEL_DIR}/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2": f"{MODEL_DIR}/microsoft-phi-2",
    "Gemma-2B": f"{MODEL_DIR}/google-gemma-2b"
}

# Load evaluation metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Load dataset
mlm_dataset = load_dataset("glue", "sst2", split="test[:50]")
clm_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
sample_texts = clm_dataset["text"][:5]


def evaluate_mlm(model_name):
    model = AutoModelForMaskedLM.from_pretrained(mlm_models[model_name], local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mlm_models[model_name], local_files_only=True)

    total_accuracy = []
    model_responses = []
    for sample in mlm_dataset:
        text = sample["sentence"]
        words = text.split()
        if len(words) < 2:
            continue
        masked_index = np.random.randint(0, len(words))
        masked_text = words[:masked_index] + ["[MASK]"] + words[masked_index + 1:]
        masked_text = " ".join(masked_text)
        input_ids = tokenizer(masked_text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids).logits
        
        masked_position = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
        predicted_token_id = torch.argmax(outputs[0, masked_position]).item()
        predicted_word = tokenizer.decode([predicted_token_id])
        
        is_correct = (predicted_word.lower() == words[masked_index].lower())
        total_accuracy.append(is_correct)
        model_responses.append({"Original": text, "Masked": masked_text, "Predicted": predicted_word, "Actual": words[masked_index], "Correct": is_correct})
    
    accuracy = np.mean(total_accuracy)
    return {"Accuracy": accuracy, "Responses": model_responses[:5]}


def evaluate_clm(model_name):
    model = AutoModelForCausalLM.from_pretrained(clm_models[model_name], local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(clm_models[model_name], local_files_only=True)
    
    predictions = []
    references = []
    log_likelihoods = []
    
    for text in sample_texts:
        if not text.strip():
            continue
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=200).to(device)
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
    perplexity = math.exp(-np.mean(log_likelihoods)) if log_likelihoods else float('inf')
    
    return {"BLEU Score": bleu_score, "ROUGE-1": rouge_scores.get("rouge1", 0), "ROUGE-2": rouge_scores.get("rouge2", 0), "ROUGE-L": rouge_scores.get("rougeL", 0), "Perplexity": perplexity}

@app.route('/')
def index():
    mlm_results = {name: evaluate_mlm(name) for name in mlm_models.keys()}
    clm_results = {name: evaluate_clm(name) for name in clm_models.keys()}
    return render_template("index.html", mlm_results=mlm_results, clm_results=clm_results)

if __name__ == '__main__':
    app.run(debug=True)
