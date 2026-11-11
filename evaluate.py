import json
import math
import os
import time
import torch
import tiktoken
from config import BATCH_SIZE, CONTEXT_LENGTH, DIM, N_HEADS, N_LAYERS, device
from data import get_batch, vocab_size
from model import ModernLLM

tokenizer = tiktoken.get_encoding("gpt2")

# Standardized benchmark suite with explicit protagonists for consistency tracking
BENCHMARK_PROMPTS = [
    {"prompt": "Once upon a time, there was a little dog named Max who loved to", "entity": "Max"},
    {"prompt": "One sunny morning, a little girl named Lily found a shiny red", "entity": "Lily"},
    {"prompt": "In a big green forest, a brave rabbit named Benny wanted to", "entity": "Benny"},
    {"prompt": "Tom had a favorite blue toy truck that he took to the park every", "entity": "Tom"},
    {"prompt": "Once there was a tiny kitten named Mia who was afraid of the", "entity": "Mia"},
    {"prompt": "Sam and his mom went to the kitchen to bake a delicious chocolate", "entity": "Sam"},
    {"prompt": "High up in an oak tree, a little bird named Pip learned how to", "entity": "Pip"},
    {"prompt": "Every afternoon, a cheerful bear named Barnaby looked for sweet", "entity": "Barnaby"},
]


@torch.no_grad()
def compute_perplexity(model, eval_iters=40):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = get_batch("val", BATCH_SIZE, CONTEXT_LENGTH)
        _, loss = model(X, Y)
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(min(mean_loss, 20))


def compute_ngram_metrics(texts):
    all_unigrams, all_bigrams, all_trigrams = [], [], []
    repeated_4grams, total_4grams = 0, 0

    for text in texts:
        words = text.lower().replace(".", " ").replace(",", " ").split()
        all_unigrams.extend(words)
        all_bigrams.extend(zip(words, words[1:]))
        all_trigrams.extend(zip(words, words[1:], words[2:]))

        fourgrams = list(zip(words, words[1:], words[2:], words[3:]))
        total_4grams += len(fourgrams)
        repeated_4grams += len(fourgrams) - len(set(fourgrams))

    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    distinct_3 = len(set(all_trigrams)) / max(len(all_trigrams), 1)
    rep_4gram_rate = repeated_4grams / max(total_4grams, 1)

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "distinct_3": round(distinct_3, 4),
        "repetition_4gram_rate": round(rep_4gram_rate, 4),
    }


@torch.no_grad()
def evaluate_narrative_quality(model, max_new_tokens=220):
    generated_texts = []
    entity_retained = 0
    clean_endings = 0

    for item in BENCHMARK_PROMPTS:
        prompt = item["prompt"]
        entity = item["entity"]

        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out_ids = model.generate(idx, max_new_tokens=max_new_tokens, use_cache=True)[0].tolist()

        # Trim <|endoftext|> if present
        if tokenizer.eot_token in out_ids:
            out_ids = out_ids[: out_ids.index(tokenizer.eot_token)]

        full_text = tokenizer.decode(out_ids).strip()
        generated_texts.append(full_text)

        # Check if the protagonist is still present in the second half of the story
        second_half = full_text[len(full_text) // 2 :]
        if entity.lower() in second_half.lower():
            entity_retained += 1

        if full_text.endswith((".", "!", "?", '."', '!"', '?"')):
            clean_endings += 1

    ngram_stats = compute_ngram_metrics(generated_texts)
    return {
        **ngram_stats,
        "entity_retention_rate": round(entity_retained / len(BENCHMARK_PROMPTS), 4),
        "clean_completion_rate": round(clean_endings / len(BENCHMARK_PROMPTS), 4),
        "samples": generated_texts[:2],
    }


@torch.no_grad()
def benchmark_kv_cache(model, gen_tokens=120, runs=3):
    prompt_ids = torch.tensor(
        [tokenizer.encode("Once upon a time, there was a little")],
        dtype=torch.long,
        device=device,
    )

    results = {}
    for use_cache in [False, True]:
        # Warmup pass
        _ = model.generate(prompt_ids, max_new_tokens=16, use_cache=use_cache, eot_token=None)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(runs):
            _ = model.generate(
                prompt_ids, max_new_tokens=gen_tokens, use_cache=use_cache, eot_token=None
            )
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tps = (gen_tokens * runs) / elapsed
        key = "kv_cache_on_tok_per_sec" if use_cache else "kv_cache_off_tok_per_sec"
        results[key] = round(tps, 2)

    results["speedup_multiplier"] = round(
        results["kv_cache_on_tok_per_sec"] / max(results["kv_cache_off_tok_per_sec"], 1e-5), 2
    )
    return results


if __name__ == "__main__":
    ckpt_path = os.path.join("checkpoints", "best.pt")
    model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from step {ckpt['step']}")

    val_loss, val_ppl = compute_perplexity(model)
    quality_metrics = evaluate_narrative_quality(model)
    speed_metrics = benchmark_kv_cache(model)

    report = {
        "val_loss": round(val_loss, 4),
        "val_perplexity": round(val_ppl, 2),
        **quality_metrics,
        **speed_metrics,
    }

    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== SYSTEMATIC EVALUATION REPORT ===")
    for k, v in report.items():
        if k != "samples":
            print(f"{k:28s}: {v}")
