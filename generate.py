import os
import torch
import torch.nn.functional as F
import tiktoken
from config import CONTEXT_LENGTH, DIM, N_HEADS, N_LAYERS, device
from model import ModernLLM

# ==========================================
# CHANGE YOUR PROMPT & GENERATION SETTINGS HERE
# ==========================================
PROMPT = "Once upon a time, there was a little dog named Max who loved to"
MAX_NEW_TOKENS = 150       # Target story length before looking for a period to stop
TEMPERATURE = 0.65         # Slightly lower temperature keeps plot & characters steady
TOP_K = 30                 # Hard cap on candidate tokens
TOP_P = 0.85               # Nucleus sampling: drops low-probability random tokens
REPETITION_PENALTY = 1.12  # Applied ONLY to the last 24 tokens to prevent stuttering
INTERACTIVE = False        # Set True to enter prompts in the terminal
CHECKPOINT_PATH = os.path.join("checkpoints", "best.pt")
# ==========================================

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

USE_AMP = device.type in {"mps", "cuda"}
AMP_DTYPE = torch.bfloat16


def load_trained_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        fallback = os.path.join("checkpoints", "latest.pt")
        if os.path.exists(fallback):
            print(f"'{ckpt_path}' not found. Falling back to '{fallback}'.")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(
                f"No checkpoint found at '{ckpt_path}'. Run train.py first!"
            )

    model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded '{ckpt_path}' (Trained for {checkpoint['step']} steps | "
        f"Val loss: {checkpoint['val_loss']:.4f})"
    )
    return model


@torch.no_grad()
def generate_text(
    model,
    prompt,
    max_new_tokens=150,
    temperature=0.65,
    top_k=30,
    top_p=0.85,
    repetition_penalty=1.12,
):
    input_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    stop_phrases = ["\nOnce upon a time", "\nOne day, there was", "\nInstruction:"]
    kv_caches = None

    # Allow up to 40 extra tokens after max_new_tokens so the final sentence finishes cleanly
    for step in range(max_new_tokens + 40):
        # 1. Autoregressive step using KV-cache
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            if kv_caches is not None:
                start_pos = idx.size(1) - 1
                logits, _, kv_caches = model(
                    idx[:, -1:], start_pos=start_pos, kv_caches=kv_caches, use_cache=True
                )
            else:
                logits, _, kv_caches = model(
                    idx, start_pos=0, kv_caches=None, use_cache=True
                )

        logits = logits[:, -1, :].float()

        # 2. Local repetition penalty (ONLY on recent window to preserve protagonist names)
        if repetition_penalty != 1.0:
            local_window = idx[0, -24:]
            recent_tokens = torch.unique(local_window)
            logits_for_recent = logits[0, recent_tokens]
            logits[0, recent_tokens] = torch.where(
                logits_for_recent < 0,
                logits_for_recent * repetition_penalty,
                logits_for_recent / repetition_penalty,
            )

        # 3. Temperature scaling
        logits = logits / max(temperature, 1e-5)

        # 4. Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # 5. Top-P (Nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # End immediately on EOS token
        if idx_next.item() == tokenizer.eot_token:
            break

        idx = torch.cat((idx, idx_next), dim=1)

        # Decode generated text and inspect stopping criteria
        full_text = tokenizer.decode(idx[0].tolist())
        generated_part = full_text[prompt_len:]

        # Early exit if a subsequent story starts
        for stop_phrase in stop_phrases:
            if stop_phrase in generated_part:
                cutoff = prompt_len + generated_part.index(stop_phrase)
                return full_text[:cutoff].strip()

        # Exit cleanly on sentence boundary once target length is reached
        if step >= max_new_tokens and full_text.rstrip().endswith(
            (".", "!", "?", '."', '!"', '?"')
        ):
            break

    return tokenizer.decode(idx[0].tolist()).strip()


if __name__ == "__main__":
    model = load_trained_model(CHECKPOINT_PATH)

    print("\n--- Generated Output ---")
    output = generate_text(
        model,
        prompt=PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )
    print(output)
    print("------------------------\n")

    if INTERACTIVE:
        print("Interactive mode enabled. Type a prompt and press Enter (or 'q' to quit):")
        while True:
            user_prompt = input("\nPrompt > ").strip()
            if user_prompt.lower() in {"q", "quit", "exit"}:
                break
            if not user_prompt:
                continue
            print(
                "\n"
                + generate_text(
                    model,
                    user_prompt,
                    MAX_NEW_TOKENS,
                    TEMPERATURE,
                    TOP_K,
                    TOP_P,
                    REPETITION_PENALTY,
                )
            )
