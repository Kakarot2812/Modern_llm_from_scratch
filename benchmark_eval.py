# benchmark_eval.py
import csv
import json
import os
import random
import re
import time
import torch
import tiktoken

from config import CONTEXT_LENGTH, DIM, N_HEADS, N_LAYERS, device
from model import ModernLLM
from benchmark_prompts import BENCHMARK_50

BASE_CKPT = os.path.join("checkpoints", "best.pt")
SFT_CKPT = os.path.join("checkpoints", "sft_best.pt")

# Standardized Generation Hyperparameters (Identical for both models)
MAX_NEW_TOKENS = 180
TEMPERATURE = 0.60
TOP_K = 30
TOP_P = 0.85
REPETITION_PENALTY = 1.18
CFG_SCALE = 1.0  # Strictly disabled to isolate weights

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab


def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("step", "N/A"), ckpt.get("val_loss", float("nan"))


@torch.no_grad()
def generate_story_controlled(model, prompt_text):
    """
    Both Base and SFT receive the identical formatted prompt,
    identical decoding parameters, and no CFG.
    """
    formatted = f"Instruction: {prompt_text.strip()}\nStory:\n"
    idx = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long, device=device)

    out_ids = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        use_cache=True,
        eot_token=tokenizer.eot_token,
        cfg_scale=1.0,
        uncond_idx=None,
        grace_tokens=50,
    )[0].tolist()

    if tokenizer.eot_token in out_ids:
        out_ids = out_ids[: out_ids.index(tokenizer.eot_token)]

    decoded = tokenizer.decode(out_ids).strip()
    if "Story:\n" in decoded:
        story_part = decoded.split("Story:\n", 1)[1].strip()
    else:
        story_part = decoded
    return story_part


def score_rubric_automated(story, item):
    """
    Automated proxies for comparison reference only.
    True semantic evaluation should rely on blinded human scoring.
    """
    words = re.findall(r"[a-z0-9]+", story.lower())
    if not words:
        return {k: 0.0 for k in ["char_adherence", "topic_adherence", "coherence", "char_consistency", "ending", "repetition", "total"]}

    first_sentence = re.split(r"[.!?]", story)[0].lower()
    char_lower = item["character"].lower()
    subj_lower = item["subject"].lower()

    # 1. Character Adherence (20 pts)
    char_found = char_lower in first_sentence or char_lower in story[:200].lower()
    subj_found = subj_lower in first_sentence or subj_lower in story[:200].lower()
    char_adherence = 20.0 if (char_found and subj_found) else (12.0 if (char_found or subj_found) else 0.0)

    # 2. Topic Adherence (20 pts)
    target_kws = item["keywords"]
    matched_kws = sum(1 for kw in target_kws if re.search(rf"\b{kw}\b", story.lower()))
    coverage = matched_kws / max(len(target_kws), 1)
    topic_adherence = round(min(coverage / 0.5, 1.0) * 20.0, 2)

    # 3. Narrative Coherence proxy (25 pts)
    sentences = [s.strip() for s in re.split(r"[.!?]+", story) if len(s.strip()) > 3]
    coherence = 0.0
    if len(words) >= 60:
        coherence += 10.0
    elif len(words) >= 30:
        coherence += 5.0
    if len(sentences) >= 4:
        coherence += 10.0
    elif len(sentences) >= 2:
        coherence += 5.0
    if re.match(r"^(once upon a time|one day|one sunny|there was|in a|every)", story.lower()):
        coherence += 5.0

    # 4. Character Consistency (15 pts)
    second_half = story[len(story) // 2 :].lower()
    if char_lower in second_half:
        char_consistency = 15.0
    elif subj_lower in second_half:
        char_consistency = 9.0
    else:
        char_consistency = 0.0

    # 5. Ending Quality (10 pts)
    clean_terminal = story.rstrip().endswith((".", "!", "?", '."', '!"', '?"'))
    ending = 6.0 if clean_terminal else 0.0
    closing_markers = ["from that day", "happily", "smiled", "learned", "proud", "best friends", "thanked"]
    if any(m in story[-150:].lower() for m in closing_markers):
        ending += 4.0

    # 6. Repetition (10 pts)
    fourgrams = list(zip(words, words[1:], words[2:], words[3:]))
    if fourgrams:
        rep_rate = (len(fourgrams) - len(set(fourgrams))) / len(fourgrams)
        repetition = round(max(0.0, 1.0 - (rep_rate / 0.08)) * 10.0, 2)
    else:
        repetition = 10.0

    total = round(char_adherence + topic_adherence + coherence + char_consistency + ending + repetition, 2)
    return {
        "char_adherence": char_adherence,
        "topic_adherence": topic_adherence,
        "coherence": coherence,
        "char_consistency": char_consistency,
        "ending": ending,
        "repetition": repetition,
        "total": total,
    }


if __name__ == "__main__":
    print("Initializing Controlled 50-Prompt Benchmark (Experiment A)...")
    base_model, base_step, base_loss = load_model(BASE_CKPT)
    print(f"Loaded Base Model (Step: {base_step} | Val Loss: {base_loss:.4f})")

    sft_model, sft_step, sft_loss = load_model(SFT_CKPT)
    print(f"Loaded SFT Model  (Step: {sft_step} | Val Loss: {sft_loss:.4f})")
    print("Inference Mode: Both models get identical formatted prompts | CFG = 1.0 (OFF)")
    print("=" * 80)

    base_results, sft_results = [], []
    blind_rows = []
    blind_key = {}

    rng = random.Random(42)
    start_time = time.time()

    for item in BENCHMARK_50:
        p_id = item["id"]
        cat = item["category"]
        prompt = item["prompt"]

        print(f"[{p_id:02d}/50] ({cat}) {prompt[:65]}...")

        # 1. Base Model Inference (Controlled)
        torch.manual_seed(1000 + p_id)
        base_story = generate_story_controlled(base_model, prompt)
        base_score = score_rubric_automated(base_story, item)
        base_results.append({"id": p_id, "score": base_score, "story": base_story})

        # 2. SFT Model Inference (Controlled: No CFG)
        torch.manual_seed(1000 + p_id)
        sft_story = generate_story_controlled(sft_model, prompt)
        sft_score = score_rubric_automated(sft_story, item)
        sft_results.append({"id": p_id, "score": sft_score, "story": sft_story})

        # 3. Blind assignment for Human Scoring Sheet (Experiment B)
        # Randomly assign Base and SFT to Story_A and Story_B
        swap = rng.choice([False, True])
        story_a = sft_story if swap else base_story
        story_b = base_story if swap else sft_story
        blind_key[p_id] = {"Story_A": "SFT" if swap else "Base", "Story_B": "Base" if swap else "SFT"}

        blind_rows.append({
            "Prompt_ID": p_id,
            "Category": cat,
            "Prompt": prompt,
            "Target_Character": item["character"],
            "Target_Subject": item["subject"],
            "Story_A": story_a,
            "A_CharAdh_20": "",
            "A_TopicAdh_20": "",
            "A_Coherence_25": "",
            "A_Consistency_15": "",
            "A_Ending_10": "",
            "A_Repetition_10": "",
            "A_Total_Score": "",
            "Story_B": story_b,
            "B_CharAdh_20": "",
            "B_TopicAdh_20": "",
            "B_Coherence_25": "",
            "B_Consistency_15": "",
            "B_Ending_10": "",
            "B_Repetition_10": "",
            "B_Total_Score": "",
        })

    elapsed = time.time() - start_time
    print(f"\nCompleted 50 prompt pairs in {elapsed:.1f}s ({elapsed/50:.2f}s per pair).")

    # Aggregate metric summaries
    def aggregate_scores(res_list):
        keys = ["char_adherence", "topic_adherence", "coherence", "char_consistency", "ending", "repetition", "total"]
        return {k: round(sum(r["score"][k] for r in res_list) / len(res_list), 2) for k in keys}

    base_avg = aggregate_scores(base_results)
    sft_avg = aggregate_scores(sft_results)

    print("\n" + "=" * 80)
    print(" " * 18 + "CONTROLLED 50-PROMPT REPORT (NO CFG, IDENTICAL INPUTS)")
    print("=" * 80)
    print(f"{'Rubric Dimension':<30} | {'Max Pts':<8} | {'Base Model':<12} | {'SFT Model':<12} | {'Delta':<8}")
    print("-" * 80)
    dim_names = [
        ("Character Adherence", "char_adherence", 20),
        ("Topic Adherence", "topic_adherence", 20),
        ("Narrative Coherence", "coherence", 25),
        ("Character Consistency", "char_consistency", 15),
        ("Ending Quality", "ending", 10),
        ("Repetition Rate", "repetition", 10),
        ("TOTAL COMPOSITE SCORE", "total", 100),
    ]
    for label, k, max_p in dim_names:
        diff = sft_avg[k] - base_avg[k]
        sign = "+" if diff > 0 else ""
        print(f"{label:<30} | {max_p:<8} | {base_avg[k]:<12.2f} | {sft_avg[k]:<12.2f} | {sign}{diff:<7.2f}")
    print("=" * 80)

    # Save Blinded Human Scoring Sheet
    csv_file = "blinded_human_scoring_sheet.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(blind_rows[0].keys()))
        writer.writeheader()
        writer.writerows(blind_rows)
    print(f"Saved blinded scoring sheet to: {csv_file}")

    # Save Blind Key
    with open("blind_key.json", "w", encoding="utf-8") as f:
        json.dump(blind_key, f, indent=2)
    print("Saved unblinding key to: blind_key.json (Keep this hidden until grading is complete)")

    # Save complete JSON
    json_report = {
        "experiment": "Experiment A (Controlled Comparison - No CFG, Identical Prompts)",
        "hyperparameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "repetition_penalty": REPETITION_PENALTY,
            "cfg_scale": 1.0,
        },
        "aggregate_summary": {
            "base_model": base_avg,
            "sft_model": sft_avg,
        },
    }
    with open("benchmark_controlled_results.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

