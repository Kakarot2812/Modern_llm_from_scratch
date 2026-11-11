import os
import torch
import torch.nn.functional as F
import tiktoken
from config import CONTEXT_LENGTH, DIM, N_HEADS, N_LAYERS, device
from model import ModernLLM

# Test prompts designed to contrast pretraining vs instruction tuning
TEST_PROMPTS = [
    # 1. Standard open-ended story beginnings
    "Once upon a time, there was a little dog named Max who loved to",
    "One sunny morning, a little girl named Lily found a shiny red",
    
    # 2. Instruction prompts formatted like SFT training
    "Instruction: Write a wholesome children's story with a clear beginning, middle, and ending.\nStory:\n",
    "Instruction: Write a wholesome children's story with a clear beginning, middle, and ending.\nStory:\nOnce upon a time, Benny the rabbit",
]

BASE_CKPT = os.path.join("checkpoints", "best.pt")
SFT_CKPT = os.path.join("checkpoints", "sft_best.pt")

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

USE_AMP = device.type in {"mps", "cuda"}
AMP_DTYPE = torch.bfloat16


def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")
    model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("step", "N/A"), checkpoint.get("val_loss", float("nan"))


@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_tokens=140,
    temperature=0.65,
    top_k=30,
    top_p=0.85,
    repetition_penalty=1.12,
):
    input_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    stop_phrases = ["\nOnce upon a time", "\nInstruction:"]
    kv_caches = None

    for step in range(max_new_tokens + 40):
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            if kv_caches is not None:
                start_pos = idx.size(1) - 1
                logits, _, kv_caches = model(
                    idx[:, -1:], start_pos=start_pos, kv_caches=kv_caches, use_cache=True
                )
            else:
                logits, _, kv_caches = model(idx, start_pos=0, kv_caches=None, use_cache=True)

        logits = logits[:, -1, :].float()

        # Local repetition penalty
        if repetition_penalty != 1.0:
            local_window = idx[0, -24:]
            recent_tokens = torch.unique(local_window)
            logits_for_recent = logits[0, recent_tokens]
            logits[0, recent_tokens] = torch.where(
                logits_for_recent < 0,
                logits_for_recent * repetition_penalty,
                logits_for_recent / repetition_penalty,
            )

        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() == tokenizer.eot_token:
            break

        idx = torch.cat((idx, idx_next), dim=1)

        full_text = tokenizer.decode(idx[0].tolist())
        generated_part = full_text[prompt_len:]

        for stop_phrase in stop_phrases:
            if stop_phrase in generated_part:
                cutoff = prompt_len + generated_part.index(stop_phrase)
                return full_text[:cutoff].strip()

        if step >= max_new_tokens and full_text.rstrip().endswith((".", "!", "?", '."', '!"', '?"')):
            break

    return tokenizer.decode(idx[0].tolist()).strip()


if __name__ == "__main__":
    print("Loading models for comparison...")
    base_model, base_step, base_loss = load_model(BASE_CKPT)
    print(f"Loaded Base Model  | Step: {base_step} | Val Loss: {base_loss:.4f}")

    sft_model, sft_step, sft_loss = load_model(SFT_CKPT)
    print(f"Loaded SFT Model   | Step: {sft_step} | Val Loss: {sft_loss:.4f}")

    print("\n" + "=" * 80)
    print(" " * 26 + "BASE vs SFT MODEL COMPARISON")
    print("=" * 80)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n--- [PROMPT {i}] ---")
        print(prompt.strip())
        print("-" * 80)

        # Generate from Base Model
        torch.manual_seed(42)
        base_out = generate(base_model, prompt)
        print("\n[BASE MODEL OUTPUT]:")
        print(base_out)

        print("\n" + "." * 40)

        # Generate from SFT Model
        torch.manual_seed(42)
        sft_out = generate(sft_model, prompt)
        print("\n[SFT MODEL OUTPUT]:")
        print(sft_out)

        print("\n" + "=" * 80)
