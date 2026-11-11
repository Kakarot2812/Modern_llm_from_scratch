# import glob
# import os
# import time
# import matplotlib.pyplot as plt
# import torch
# from config import (
#     BATCH_SIZE,
#     CONTEXT_LENGTH,
#     DIM,
#     EVAL_INTERVAL,
#     LR,
#     MAX_ITERS,
#     N_HEADS,
#     N_LAYERS,
#     device,
# )
# from data import decode, encode, get_batch, vocab_size
# from model import ModernLLM

# CHECKPOINT_DIR = "checkpoints"
# KEEP_LAST_N = 3  # Number of periodic step checkpoints to retain on disk
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


# def atomic_save(obj, path):
#     """Write to a temp file first so Ctrl+C never corrupts a checkpoint."""
#     tmp_path = f"{path}.tmp"
#     torch.save(obj, tmp_path)
#     os.replace(tmp_path, path)


# def save_checkpoint(
#     step, val_loss, best_val_loss, train_losses, val_losses, eval_steps, is_best=False
# ):
#     checkpoint = {
#         "step": step,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "val_loss": val_loss,
#         "best_val_loss": best_val_loss,
#         "train_losses": train_losses,
#         "val_losses": val_losses,
#         "eval_steps": eval_steps,
#         "rng_state": torch.get_rng_state(),
#     }

#     # 1. Save latest checkpoint (for seamless resuming)
#     atomic_save(checkpoint, os.path.join(CHECKPOINT_DIR, "latest.pt"))

#     # 2. Save numbered step checkpoint and prune older ones
#     step_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step:06d}.pt")
#     atomic_save(checkpoint, step_path)

#     step_files = sorted(
#         glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_step_*.pt"))
#     )
#     for old_file in step_files[:-KEEP_LAST_N]:
#         os.remove(old_file)

#     # 3. Save best checkpoint separately
#     if is_best:
#         atomic_save(checkpoint, os.path.join(CHECKPOINT_DIR, "best.pt"))

#     print(
#         f"Checkpoint saved at step {step} "
#         f"{'(new best!)' if is_best else ''}"
#     )


# def load_checkpoint(path):
#     checkpoint = torch.load(path, map_location=device, weights_only=False)

#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#     if "rng_state" in checkpoint:
#         torch.set_rng_state(checkpoint["rng_state"].cpu())

#     print(
#         f"Resumed from step {checkpoint['step']} "
#         f"| Val loss: {checkpoint['val_loss']:.4f} "
#         f"| Best val loss: {checkpoint.get('best_val_loss', checkpoint['val_loss']):.4f}"
#     )
#     return checkpoint


# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ["train", "val"]:
#         losses = torch.zeros(50)
#         for k in range(50):
#             X, Y = get_batch(split, BATCH_SIZE, CONTEXT_LENGTH)
#             _, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out


# # --- Training State Initialization ---

# train_losses, val_losses, eval_steps = [], [], []
# best_val_loss = float("inf")
# start_step = 0

# latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
# if os.path.exists(latest_path):
#     ckpt = load_checkpoint(latest_path)
#     start_step = ckpt["step"] + 1
#     best_val_loss = ckpt.get("best_val_loss", ckpt["val_loss"])
#     train_losses = ckpt.get("train_losses", [])
#     val_losses = ckpt.get("val_losses", [])
#     eval_steps = ckpt.get("eval_steps", [])

# # --- Training Loop ---

# start_time = time.time()

# for step in range(start_step, MAX_ITERS):
#     # 1. Forward & backward pass first so `step` is completed before saving
#     xb, yb = get_batch("train", BATCH_SIZE, CONTEXT_LENGTH)
#     _, loss = model(xb, yb)

#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

#     # 2. Evaluate & checkpoint at intervals and on the final step
#     is_eval_step = (step % EVAL_INTERVAL == 0) or (step == MAX_ITERS - 1)
#     if is_eval_step:
#         losses = estimate_loss()
#         train_loss = losses["train"].item()
#         val_loss = losses["val"].item()

#         print(
#             f"Step {step} | "
#             f"Train loss: {train_loss:.4f} | "
#             f"Val loss: {val_loss:.4f} | "
#             f"Time: {time.time() - start_time:.1f}s"
#         )

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
#         eval_steps.append(step)

#         is_best = val_loss < best_val_loss
#         if is_best:
#             best_val_loss = val_loss

#         save_checkpoint(
#             step=step,
#             val_loss=val_loss,
#             best_val_loss=best_val_loss,
#             train_losses=train_losses,
#             val_losses=val_losses,
#             eval_steps=eval_steps,
#             is_best=is_best,
#         )

# # --- Plot Losses ---

# plt.figure(figsize=(8, 5))
# plt.plot(eval_steps, train_losses, label="train")
# plt.plot(eval_steps, val_losses, label="val")
# plt.xlabel("Step")
# plt.ylabel("Loss")
# plt.legend()
# plt.savefig("loss.png")
# plt.close()

# # --- Generate Sample Text Using Best Model ---

# best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
# if os.path.exists(best_path):
#     load_checkpoint(best_path)

# model.eval()
# context = torch.tensor([encode("ROMEO:\n")], dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))


import glob
import math
import os
import shutil
import time
import matplotlib.pyplot as plt
import torch
from config import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    DIM,
    EVAL_INTERVAL,
    GRAD_CLIP,
    LR,
    MAX_ITERS,
    MIN_LR,
    N_HEADS,
    N_LAYERS,
    WARMUP_ITERS,
    device,
)
from data import decode, encode, get_batch, vocab_size
from model import ModernLLM

CHECKPOINT_DIR = "checkpoints"
KEEP_LAST_N = 3
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

# Enable bfloat16 mixed precision on MPS / CUDA to halve memory and speed up training
use_amp = device.type in {"mps", "cuda"}
amp_dtype = torch.bfloat16


def get_lr(step):
    if step < WARMUP_ITERS:
        return LR * (step + 1) / WARMUP_ITERS
    if step >= MAX_ITERS:
        return MIN_LR
    decay_ratio = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


def atomic_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(step, val_loss, best_val_loss, train_losses, val_losses, eval_steps, is_best=False):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "eval_steps": eval_steps,
        "rng_state": torch.get_rng_state(),
    }

    latest_file = os.path.join(CHECKPOINT_DIR, "latest.pt")
    atomic_save(checkpoint, latest_file)

    step_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step:06d}.pt")
    shutil.copyfile(latest_file, step_path)

    step_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_step_*.pt")))
    for old_file in step_files[:-KEEP_LAST_N]:
        os.remove(old_file)

    if is_best:
        shutil.copyfile(latest_file, os.path.join(CHECKPOINT_DIR, "best.pt"))

    print(f"Checkpoint saved at step {step} {'(new best!)' if is_best else ''}")


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    return checkpoint


@torch.no_grad()
def estimate_loss(eval_iters=20):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, BATCH_SIZE, CONTEXT_LENGTH)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":
    train_losses, val_losses, eval_steps = [], [], []
    best_val_loss = float("inf")
    start_step = 0

    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.exists(latest_path):
        ckpt = load_checkpoint(latest_path)
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt["val_loss"])
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        eval_steps = ckpt.get("eval_steps", [])

    start_time = time.time()

    for step in range(start_step, MAX_ITERS):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb = get_batch("train", BATCH_SIZE, CONTEXT_LENGTH)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if (step % EVAL_INTERVAL == 0) or (step == MAX_ITERS - 1):
            losses = estimate_loss()
            train_loss = losses["train"].item()
            val_loss = losses["val"].item()
            ppl = math.exp(min(val_loss, 20))

            print(
                f"Step {step:4d} | LR: {lr:.2e} | Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | PPL: {ppl:.2f} | Time: {time.time() - start_time:.1f}s"
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            eval_steps.append(step)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            save_checkpoint(
                step, val_loss, best_val_loss, train_losses, val_losses, eval_steps, is_best
            )

            if device.type == "mps":
                torch.mps.empty_cache()

    plt.figure(figsize=(8, 5))
    plt.plot(eval_steps, train_losses, label="Train Loss")
    plt.plot(eval_steps, val_losses, label="Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("ModernLLM Training Curve")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()

    best_path = os.path.join(CHECKPOINT_DIR, "best.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path)

    model.eval()
    context = torch.tensor([encode("Once upon a time, ")], dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=350)[0].tolist()))
