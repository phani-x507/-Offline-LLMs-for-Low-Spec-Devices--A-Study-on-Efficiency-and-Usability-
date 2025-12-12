import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")


MODEL_DIR = "models_path"


mlm_models = {
    "DistilBERT": f"{MODEL_DIR}/distilbert-base-uncased",
    "MobileBERT": f"{MODEL_DIR}/google-mobilebert-uncased"
}


accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


dataset = load_dataset("glue", "sst2", split="test[:50]")


def evaluate_mlm(model_name, dataset):
    print(f"\n🔹 Evaluating {model_name}...")


    model = AutoModelForMaskedLM.from_pretrained(mlm_models[model_name], local_files_only=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mlm_models[model_name], local_files_only=True)

    total_accuracy = []
    total_f1 = []
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

    print(f"✅ Results for {model_name}: Accuracy: {accuracy:.4f}\n")


    print("📌 Sample Predictions:")
    for i, resp in enumerate(model_responses[:5]):
        print(f"{i+1}. Original: {resp['Original']}\n   Masked: {resp['Masked']}\n   Predicted: {resp['Predicted']} (Actual: {resp['Actual']}) - {'✅' if resp['Correct'] else '❌'}\n")

    return {"Accuracy": accuracy, "Responses": model_responses}


results = {}
for name in mlm_models:
    results[name] = evaluate_mlm(name, dataset)
