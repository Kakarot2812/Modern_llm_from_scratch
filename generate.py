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
MAX_NEW_TOKENS = 110       # Target story length before looking for a period to stop
TEMPERATURE = 0.65         # Slightly lower temperature keeps plot & characters steady
TOP_K = 30                 # Hard cap on candidate tokens
TOP_P = 0.85               # Nucleus sampling: drops low-probability random names
REPETITION_PENALTY = 1.12  # Applied ONLY to the last 16 tokens to prevent stuttering
INTERACTIVE = False        # Set True to enter prompts in the terminal
CHECKPOINT_PATH = os.path.join("checkpoints", "best.pt")
# ==========================================

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab


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
    max_new_tokens=110,
    temperature=0.65,
    top_k=30,
    top_p=0.85,
    repetition_penalty=1.12,
):
    input_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    stop_phrases = ["\nOnce upon a time", "\nOne day, there was"]

    # Allow up to 35 extra tokens after max_new_tokens so the final sentence finishes
    for step in range(max_new_tokens + 35):
        idx_cond = idx[:, -model.context_length :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # 1. Local repetition penalty (ONLY last 16 tokens so main character names stay active)
        if repetition_penalty != 1.0:
            local_window = idx_cond[0, -16:]
            recent_tokens = torch.unique(local_window)
            logits_for_recent = logits[0, recent_tokens]
            logits[0, recent_tokens] = torch.where(
                logits_for_recent < 0,
                logits_for_recent * repetition_penalty,
                logits_for_recent / repetition_penalty,
            )

        # 2. Temperature scaling
        logits = logits / max(temperature, 1e-5)

        # 3. Top-K filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # 4. Top-P (Nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least the single most likely token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() == tokenizer.eot_token:
            break

        idx = torch.cat((idx, idx_next), dim=1)

        full_text = tokenizer.decode(idx[0].tolist())
        generated_part = full_text[prompt_len:]

        # Stop if a second story starts
        for stop_phrase in stop_phrases:
            if stop_phrase in generated_part:
                cutoff = prompt_len + generated_part.index(stop_phrase)
                return full_text[:cutoff].strip()

        # Stop at the end of a sentence once we pass max_new_tokens
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