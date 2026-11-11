import os
import torch
import tiktoken
from config import CONTEXT_LENGTH, DIM, N_HEADS, N_LAYERS, device
from model import ModernLLM

TEST_PROMPTS = [
    "Instruction: Write a story about a little robot named Max who wants to cook and make yummy food.\nStory:\n",
    "Instruction: Write a story about a brave rabbit named Benny who goes on an adventure at the park.\nStory:\n",
    "Instruction: Write a story about a curious kitten named Mia who learns the importance of helping others.\nStory:\n",
    "Instruction: Write a story about a little bird named Pip who wants to fly high in the sky.\nStory:\n",
]

UNCOND_PROMPT = "Instruction: Write a short story.\nStory:\n"
BASE_CKPT = os.path.join("checkpoints", "best.pt")
SFT_CKPT = os.path.join("checkpoints", "sft_best.pt")

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab


def load_model(ckpt_path):
    model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("step", "N/A"), checkpoint.get("val_loss", float("nan"))


@torch.no_grad()
def run_generation(model, prompt, is_sft=False):
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    uncond_idx = (
        torch.tensor([tokenizer.encode(UNCOND_PROMPT)], dtype=torch.long, device=device)
        if is_sft
        else None
    )
    out_ids = model.generate(
        idx,
        max_new_tokens=200,
        temperature=0.60,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.18,
        use_cache=True,
        eot_token=tokenizer.eot_token,
        cfg_scale=1.75 if is_sft else 1.0,
        uncond_idx=uncond_idx,
        grace_tokens=50,
    )[0].tolist()

    if tokenizer.eot_token in out_ids:
        out_ids = out_ids[: out_ids.index(tokenizer.eot_token)]
    return tokenizer.decode(out_ids).strip()


if __name__ == "__main__":
    base_model, base_step, base_loss = load_model(BASE_CKPT)
    sft_model, sft_step, sft_loss = load_model(SFT_CKPT)

    print(f"Loaded Base Model | Step: {base_step} | Val Loss: {base_loss:.4f}")
    print(f"Loaded SFT Model  | Step: {sft_step} | Val Loss: {sft_loss:.4f}")
    print("=" * 80)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n--- [PROMPT {i}] ---")
        print(prompt.strip())
        print("-" * 80)

        torch.manual_seed(42)
        print("\n[BASE MODEL OUTPUT]:")
        print(run_generation(base_model, prompt, is_sft=False))

        print("\n" + "." * 40)

        torch.manual_seed(42)
        print("\n[SFT MODEL OUTPUT]:")
        print(run_generation(sft_model, prompt, is_sft=True))
        print("\n" + "=" * 80)
