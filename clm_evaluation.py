# safe_clm_eval.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import load
import numpy as np
import math
import random
import gc
import time

# ---------- Config ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

MODEL_DIR = "models_offline"

clm_models = {
    "TinyLLaMA": f"{MODEL_DIR}/TinyLlama-1.1B-Chat-v1.0",
    # "Phi-2": f"{MODEL_DIR}/microsoft-phi-2",
     "Gemma-2B": f"{MODEL_DIR}/google-gemma-2b"
}

# metric loaders
bleu_metric = load("bleu")
rouge_metric = load("rouge")

# ---------- Prepare randomized sample texts ----------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# filter: require at least 8 words and at least one sentence punctuation
filtered_texts = [t for t in dataset["text"] if isinstance(t, str) and len(t.split()) > 8 and ("." in t or "?" in t or "!" in t)]

if len(filtered_texts) < 5:
    raise RuntimeError("Not enough valid sentences in dataset after filtering.")


def get_random_samples(n=5):
    return random.sample(filtered_texts, n)

# ---------- utility: safe model unloading ----------
def unload_model(model):
    try:
        # move to CPU and delete
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------- perplexity helper (normalized) ----------
def calculate_perplexity_from_token_logprobs(token_log_probs):
    """
    token_log_probs: list of arrays/lists of token log probabilities (natural log)
    We'll concatenate and compute perplexity = exp(- mean(log_prob_per_token))
    """
    if not token_log_probs:
        return float("inf")
    all_logs = np.concatenate([np.asarray(x).ravel() for x in token_log_probs])
    # if empty, return inf
    if all_logs.size == 0:
        return float("inf")
    avg_log_prob = np.mean(all_logs)  # average log prob per token
    ppl = math.exp(-avg_log_prob)
    return ppl

# ---------- Evaluate single CLM ----------
def evaluate_clm(model_name, sample_texts, max_input_length=200, max_new_tokens=50):
    print(f"\n🔹 Evaluating {model_name}...")
    model_path = clm_models[model_name]

    # load tokenizer and model (try GPU, fallback to CPU)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"Tokenizer load failed for {model_name}: {e}")

    # Ensure pad token exists for batch/tokenization ops
    if tokenizer.pad_token is None:
        # safest: set pad_token to eos_token if available
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.get("pad_token", "<pad>"))

    model = None
    used_device = device
    try:
        # attempt to load on GPU if available
        if device == "cuda":
            try:
                model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to("cuda")
                used_device = "cuda"
            except RuntimeError as e:
                # OOM or other runtime: fallback to CPU
                print(f"⚠️ GPU load failed for {model_name}: {e}\nFalling back to CPU for this model.")
                unload_model(model) if model is not None else None
                model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to("cpu")
                used_device = "cpu"
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to("cpu")
            used_device = "cpu"
    except Exception as e:
        raise RuntimeError(f"Model load failed for {model_name}: {e}")

    # safety: set pad_token_id in model config if missing
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None and getattr(model.config, "eos_token_id", None) is not None:
        model.config.pad_token_id = model.config.eos_token_id

    predictions = []
    references = []
    token_logprob_list = []  # list of per-sample per-token logprobs

    for idx, text in enumerate(sample_texts):
        text = text.strip()
        if not text:
            print(f"🚨 Skipping empty input at index {idx}")
            continue

        # Tokenize with truncation and return attention_mask
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length, padding=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        # move tensors to device if model is on cuda
        if used_device == "cuda":
            input_ids = input_ids.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()

        # guard: if input longer than model max length, tokenizer truncated it; generate uses input length
        input_len = input_ids.shape[1]
        if input_len == 0:
            print(f"🚨 Skipping due to zero-length tokenized input for sample {idx}")
            continue

        # generation: pass attention_mask for reliable generation
        try:
            gen_out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic; change if you want diversity
                pad_token_id=model.config.pad_token_id if hasattr(model.config, "pad_token_id") else tokenizer.eos_token_id
            )
        except Exception as e:
            # if GPU OOM or generation error, try on CPU
            print(f"⚠️ Generation failed on {used_device} for sample {idx}: {e}")
            if used_device == "cuda":
                # move model to CPU and retry once
                try:
                    model.to("cpu")
                    if input_ids.is_cuda:
                        input_ids = input_ids.cpu()
                    if attention_mask is not None and attention_mask.is_cuda:
                        attention_mask = attention_mask.cpu()
                    used_device = "cpu"
                    gen_out = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=model.config.pad_token_id if hasattr(model.config, "pad_token_id") else tokenizer.eos_token_id
                    )
                except Exception as e2:
                    print(f"❌ Generation still failed on CPU: {e2}. Skipping sample {idx}.")
                    continue
            else:
                print("❌ Generation failed on CPU, skipping sample.")
                continue

        # decode generated
        generated_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        predictions.append(generated_text)
        references.append([text])

        # Compute token log-probabilities of the input tokens under the model (per-token)
        # We'll run the model in eval mode and compute log_softmax over logits
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # outputs.logits: (batch, seq_len, vocab)
            # logits = outputs.logits  # tensor
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            log_probs = torch.log_softmax(shift_logits, dim=-1) # (batch, seq_len, vocab)
            # gather the log prob of the actual input_ids tokens
            # input_ids shape (batch, seq_len), we gather along vocab dim
            token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
            # convert to numpy (batch=1)
            token_log_probs_np = token_log_probs[0].cpu().numpy()
            token_logprob_list.append(token_log_probs_np)

        # console preview
        print(f"\n📝 Input (truncated to 120 chars): {text[:120]}...")
        print(f"🤖 {model_name} Response (truncated): {generated_text[:200]}")

    # compute metrics
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)["bleu"] if predictions else 0.0
    rouge_scores = rouge_metric.compute(predictions=predictions, references=[r[0] for r in references]) if predictions else {}
    perplexity = calculate_perplexity_from_token_logprobs(token_logprob_list) if token_logprob_list else float("inf")

    # unload model to free memory
    try:
        unload_model(model)
    except Exception:
        pass

    # return structured result
    return {
        "Model": model_name,
        "BLEU Score": float(bleu_score),
        "ROUGE-1": float(rouge_scores.get("rouge1", 0.0)),
        "ROUGE-2": float(rouge_scores.get("rouge2", 0.0)),
        "ROUGE-L": float(rouge_scores.get("rougeL", 0.0)),
        "Perplexity": float(perplexity),
        "prompts": sample_texts,
        "responses": predictions
    }

# ---------- Evaluate all models ----------
def evaluate_all_clms(n_samples=5):
    # re-sample each call to get different random inputs
    sample_texts = get_random_samples(n_samples)
    results = {}
    for name in clm_models.keys():
        try:
            results[name] = evaluate_clm(name, sample_texts)
        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")
            results[name] = {"Model": name, "error": str(e)}
    return results

# ---------- quick run ----------
if __name__ == "__main__":
    start = time.time()
    results = evaluate_all_clms(n_samples=5)
    total_time = time.time() - start
    print("\n📊 Final Results (all CLMs):")
    for m, res in results.items():
        print(f"\n=== {m} ===")
        for k, v in res.items():
            if k in ("prompts", "responses"):
                print(f"{k}: (count {len(v)})")
            else:
                print(f"{k}: {v}")
    print(f"\nTotal elapsed time: {total_time:.2f}s")
