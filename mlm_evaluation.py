import torch
import time
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import gc

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

MODEL_DIR = "models_offline"

# MLM models (local paths)
mlm_models = {
    "DistilBERT": f"{MODEL_DIR}/distilbert-base-uncased",
    "MobileBERT": f"{MODEL_DIR}/google-mobilebert-uncased"
}

# (optional) keep metric loader if you want (not used for simple boolean correctness)
# accuracy_metric = evaluate.load("accuracy")

# Load dataset once (you had test[:50])
dataset = load_dataset("glue", "sst2", split="test[:50]")

def evaluate_mlm(model_name, dataset):
    """
    Evaluate a single Masked Language Model (MLM).
    Returns: dict {"Model": model_name, "Accuracy": accuracy, "Responses": [...]}
    """
    model_path = mlm_models[model_name]
    start_time = time.time()
    print(f"\n🔹 Evaluating {model_name} from {model_path} ...")

    # Load tokenizer & model locally (one at a time)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        print(f"⚠️ Failed to load tokenizer for {model_name}: {e}")
        return {"Model": model_name, "Accuracy": 0.0, "Responses": [], "Error": str(e)}

    # Ensure tokenizer has mask token
    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        # most MLM tokenizers have [MASK], but if missing, bail out gracefully
        print(f"⚠️ Tokenizer for {model_name} has no mask token. Skipping.")
        return {"Model": model_name, "Accuracy": 0.0, "Responses": [], "Error": "no mask token"}

    try:
        model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        print(f"⚠️ Failed to load model for {model_name}: {e}")
        return {"Model": model_name, "Accuracy": 0.0, "Responses": [], "Error": str(e)}

    # Move model to device
    try:
        model.to(device)
    except Exception as e:
        print(f"⚠️ Failed to move model to device ({device}): {e}")
        # if GPU move fails, fall back to CPU
        device_cpu = "cpu"
        model.to(device_cpu)
        print("↩️ Falling back to CPU for this model.")
    
    total_accuracy = []
    model_responses = []

    # Evaluate each sample (you used random masking per sentence)
    for sample in dataset:
        text = sample.get("sentence", "")
        words = text.split()
        if len(words) < 2:
            continue  # skip trivial examples

        # choose an index to mask (random)
        masked_index = np.random.randint(0, len(words))
        # construct masked text using tokenizer.mask_token (safer)
        masked_words = words[:masked_index] + [tokenizer.mask_token] + words[masked_index + 1:]
        masked_text = " ".join(masked_words)

        # tokenize and move to device
        inputs = tokenizer(masked_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        # forward pass (no grad)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # shape: (batch, seq_len, vocab_size)

        # find mask position(s) in input_ids
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)

        # this should have at least one mask position; we masked exactly one token
        if mask_positions[1].numel() == 0:
            # fallback: try to find tokenizer.mask_token in tokenized text string
            print(f"⚠️ No mask token found in tokenized input for sample: {masked_text}")
            continue

        # we take the first mask position if multiple
        mask_pos = int(mask_positions[1].view(-1)[0].item())

        # get predicted token id at mask position
        predicted_token_id = int(torch.argmax(logits[0, mask_pos, :]).item())
        predicted_word = tokenizer.decode([predicted_token_id]).strip()

        actual_word = words[masked_index]
        is_correct = (predicted_word.lower() == actual_word.lower())
        total_accuracy.append(is_correct)

        model_responses.append({
            "Original": text,
            "Masked": masked_text,
            "Predicted": predicted_word,
            "Actual": actual_word,
            "Correct": bool(is_correct)
        })

    # compute accuracy (mean of booleans)
    accuracy = float(np.mean(total_accuracy)) if total_accuracy else 0.0

    # cleanup: free model from GPU/CPU memory
    try:
        del model
        del logits
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass

    elapsed = time.time() - start_time
    print(f"✅ Done {model_name} — Accuracy: {accuracy:.4f}, Time: {elapsed:.2f}s")

    return {"Model": model_name, "Accuracy": accuracy, "Responses": model_responses[:5]}


def evaluate_all_mlms():
    """Evaluate all configured MLMs and return dict results."""
    results = {}
    for name in mlm_models:
        results[name] = evaluate_mlm(name, dataset)
    return results


if __name__ == "__main__":
    # quick test run if executed directly
    out = evaluate_all_mlms()
    for k, v in out.items():
        print(f"\nModel: {k}\nAccuracy: {v.get('Accuracy')}\nSample responses:")
        for r in v.get("Responses", []):
            print(f" - Orig: {r['Original']}\n   Masked: {r['Masked']}\n   Pred: {r['Predicted']} | Actual: {r['Actual']} | Correct: {r['Correct']}")
