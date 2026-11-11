import os
import torch
import tiktoken
from config import CONTEXT_LENGTH, DIM, N_HEADS, N_LAYERS, device
from model import ModernLLM

# ==========================================
# CHANGE YOUR PROMPT & GENERATION SETTINGS HERE
# ==========================================
PROMPT = "Write a story about a curious kitten named Mia who learns the importance of helping others."
MAX_NEW_TOKENS = 200       # Target story length before looking for a period to stop
TEMPERATURE = 0.58         # Slightly lower temperature keeps plot & characters steady
TOP_K = 30                 # Hard cap on candidate tokens
TOP_P = 0.85               # Nucleus sampling: drops low-probability random tokens
REPETITION_PENALTY = 1.18  # Applied ONLY to recent tokens to prevent loops
CFG_SCALE = 1.75           # Prompt guidance scale for SFT checkpoint (1.0 disables CFG)
INTERACTIVE = False        # Set True to enter prompts in the terminal
CHECKPOINT_PATH = os.path.join("checkpoints", "sft_best.pt")
# ==========================================

UNCOND_PROMPT = "Instruction: Write a short story.\nStory:\n"
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab


def load_trained_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        fallback = os.path.join("checkpoints", "best.pt")
        if os.path.exists(fallback):
            print(f"'{ckpt_path}' not found. Falling back to '{fallback}'.")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(
                f"No checkpoint found at '{ckpt_path}'. Run train.py / sft.py first!"
            )

    model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"Loaded '{ckpt_path}' (Trained for {checkpoint.get('step', 'N/A')} steps | "
        f"Val loss: {checkpoint.get('val_loss', 0.0):.4f})"
    )
    return model


@torch.no_grad()
def generate_text(
    model,
    prompt,
    max_new_tokens=200,
    temperature=0.65,
    top_k=30,
    top_p=0.85,
    repetition_penalty=1.12,
    cfg_scale=1.75,
):
    is_sft = "sft" in CHECKPOINT_PATH.lower()
    if is_sft and not prompt.startswith("Instruction:"):
        formatted_prompt = f"Instruction: {prompt.strip()}\nStory:\n"
    else:
        formatted_prompt = prompt

    idx = torch.tensor([tokenizer.encode(formatted_prompt)], dtype=torch.long, device=device)
    uncond_idx = (
        torch.tensor([tokenizer.encode(UNCOND_PROMPT)], dtype=torch.long, device=device)
        if (is_sft and cfg_scale > 1.0)
        else None
    )

    out_ids = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        use_cache=True,
        eot_token=tokenizer.eot_token,
        cfg_scale=cfg_scale if is_sft else 1.0,
        uncond_idx=uncond_idx,
        grace_tokens=50,
    )[0].tolist()

    if tokenizer.eot_token in out_ids:
        out_ids = out_ids[: out_ids.index(tokenizer.eot_token)]

    return tokenizer.decode(out_ids).strip()


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
        cfg_scale=CFG_SCALE,
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
                    CFG_SCALE,
                )
            )
