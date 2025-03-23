import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "models_offline"

mlm_models = {
    "DistilBERT": f"{MODEL_DIR}/distilbert-base-uncased",
    "MobileBERT": f"{MODEL_DIR}/google-mobilebert-uncased"
}

accuracy_metric = evaluate.load("accuracy")

dataset = load_dataset("glue", "sst2", split="test[:50]")

def evaluate_mlm(model_name, dataset):
    model = AutoModelForMaskedLM.from_pretrained(mlm_models[model_name], local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mlm_models[model_name], local_files_only=True)

    total_accuracy = []
    model_responses = []

    for sample in dataset:
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

        actual_word = words[masked_index]
        is_correct = (predicted_word.lower() == actual_word.lower())
        total_accuracy.append(is_correct)

        model_responses.append({
            "Original": text,
            "Masked": masked_text,
            "Predicted": predicted_word,
            "Actual": actual_word,
            "Correct": is_correct
        })

    accuracy = np.mean(total_accuracy)

    return {"Model": model_name, "Accuracy": accuracy, "Responses": model_responses[:5]}

def evaluate_all_mlms():
    return {name: evaluate_mlm(name, dataset) for name in mlm_models}
