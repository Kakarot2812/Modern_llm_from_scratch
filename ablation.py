import gc
import math
import time
import matplotlib.pyplot as plt
import torch
from config import DIM, LR, N_HEADS, N_LAYERS, device
from data import get_batch, vocab_size
from model import ModernLLM

ABLATION_STEPS = 1000
EVAL_EVERY = 200

# Lighter batch & context for fast, lag-free ablation on MacBook Air (MPS)
AB_BATCH_SIZE = 16
AB_CONTEXT_LENGTH = 256

EXPERIMENTS = [
    {
        "name": "1. GPT-2 Baseline (LearnedPos + LayerNorm + GELU)",
        "kwargs": {"use_rope": False, "norm_type": "layernorm", "ffn_type": "gelu", "tie_weights": False},
    },
    {
        "name": "2. ModernLLM Untied (RoPE + RMSNorm + SwiGLU)",
        "kwargs": {"use_rope": True, "norm_type": "rmsnorm", "ffn_type": "swiglu", "tie_weights": False},
    },
    {
        "name": "3. ModernLLM + Tied Weights (Ours)",
        "kwargs": {"use_rope": True, "norm_type": "rmsnorm", "ffn_type": "swiglu", "tie_weights": True},
    },
]

# Enable bfloat16 autocast on MPS/CUDA to halve memory and double speed
use_amp = device.type in {"mps", "cuda"}
amp_dtype = torch.bfloat16


@torch.no_grad()
def eval_val_loss(model, iters=10):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        X, Y = get_batch("val", AB_BATCH_SIZE, AB_CONTEXT_LENGTH)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


if __name__ == "__main__":
    results = {}
    plt.figure(figsize=(9, 5))

    for exp in EXPERIMENTS:
        torch.manual_seed(42)
        name = exp["name"]
        model = ModernLLM(
            vocab_size, DIM, N_HEADS, N_LAYERS, AB_CONTEXT_LENGTH, **exp["kwargs"]
        ).to(device)
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

        steps, val_losses = [], []
        start = time.time()
        print(f"\nRunning {name} ({num_params:.2f}M params)...")

        for step in range(ABLATION_STEPS + 1):
            if step % EVAL_EVERY == 0:
                v_loss = eval_val_loss(model)
                steps.append(step)
                val_losses.append(v_loss)
                print(
                    f"  Step {step:4d} | Val Loss: {v_loss:.4f} | "
                    f"PPL: {math.exp(min(v_loss, 20)):.2f} | "
                    f"Elapsed: {time.time() - start:.1f}s"
                )

            if step < ABLATION_STEPS:
                xb, yb = get_batch("train", AB_BATCH_SIZE, AB_CONTEXT_LENGTH)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        elapsed = time.time() - start
        results[name] = {
            "params_M": round(num_params, 2),
            "final_val_loss": round(val_losses[-1], 4),
            "final_ppl": round(math.exp(val_losses[-1]), 2),
            "time_sec": round(elapsed, 1),
        }
        plt.plot(steps, val_losses, marker="o", label=f"{name} ({num_params:.1f}M)")

        # Free Unified Memory before starting the next experiment
        del model, optimizer
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.title("Architecture Ablation Study on TinyStories")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("ablation_results.png")
    plt.close()

    print("\n### Ablation Summary Table (Copy into README.md)\n")
    print("| Architecture | Params (M) | Final Val Loss | Perplexity | Time (s) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for name, m in results.items():
        print(f"| {name} | {m['params_M']}M | {m['final_val_loss']} | {m['final_ppl']} | {m['time_sec']}s |")

