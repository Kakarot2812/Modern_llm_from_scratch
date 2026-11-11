
import os
import random
import torch
from datasets import load_dataset

from config import CONTEXT_LENGTH, DIM, N_HEADS, N_LAYERS, device
from data import EOT_TOKEN, decode, encode, vocab_size
from model import ModernLLM


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SFT_STEPS = 600
SFT_LR = 2e-5
SFT_BATCH_SIZE = 16

EVAL_INTERVAL = 50
EVAL_BATCHES = 10

CHECKPOINT_DIR = "checkpoints"
BASE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")
BEST_SFT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sft_best.pt")
LAST_SFT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sft_last.pt")

# One consistent instruction for the TinyStories dataset.
SFT_PROMPT = (
    "Instruction: Write a wholesome children's story "
    "with a clear beginning, middle, and ending.\n"
    "Story:\n"
)

USE_AMP = device.type in {"mps", "cuda"}
AMP_DTYPE = torch.bfloat16


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET PREPARATION
# ============================================================

def load_sft_dataset():

    dataset = load_dataset("roneneldan/TinyStories")

    if "validation" in dataset:
        train_split = dataset["train"]
        val_split = dataset["validation"]

    else:
        # Create a validation split if the dataset
        # does not provide one.
        split = dataset["train"].train_test_split(
            test_size=0.01,
            seed=SEED
        )

        train_split = split["train"]
        val_split = split["test"]

    print(f"Training examples: {len(train_split):,}")
    print(f"Validation examples: {len(val_split):,}")

    return train_split, val_split


# ============================================================
# BATCH CONSTRUCTION
# ============================================================

def build_sft_batch(dataset_split, batch_size, context_length, rng):

    input_batch = []
    target_batch = []

    prompt_ids = encode(SFT_PROMPT)

    available_story_tokens = context_length - len(prompt_ids)

    if available_story_tokens < 1:
        raise ValueError("Prompt is longer than the model context.")

    attempts = 0
    max_attempts = batch_size * 20

    while len(input_batch) < batch_size:

        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Unable to construct a complete SFT batch. "
                "Check the dataset."
            )

        idx = rng.randrange(len(dataset_split))

        story = dataset_split[idx]["text"].strip()

        if not story:
            continue

        story_ids = encode(story)

        if not story_ids:
            continue

        # Reserve one token for EOS.
        story_ids = story_ids[:available_story_tokens]

        full_ids = prompt_ids + story_ids + [EOT_TOKEN]

        x = full_ids[:-1]
        y = full_ids[1:]

        # Prompt tokens are excluded from the loss.
        prompt_cutoff = len(prompt_ids) - 1

        y_masked = [
            -100 if i < prompt_cutoff else token
            for i, token in enumerate(y)
        ]

        # Pad to the model's context length.
        pad_len = context_length - len(x)

        if pad_len > 0:

            x = x + [EOT_TOKEN] * pad_len

            y_masked = y_masked + [-100] * pad_len

        input_batch.append(
            torch.tensor(x, dtype=torch.long)
        )

        target_batch.append(
            torch.tensor(y_masked, dtype=torch.long)
        )

    xb = torch.stack(input_batch).to(device)
    yb = torch.stack(target_batch).to(device)

    return xb, yb


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_sft(model, val_split, rng):

    was_training = model.training
    model.eval()

    total_loss = 0.0

    for _ in range(EVAL_BATCHES):

        xb, yb = build_sft_batch(
            val_split,
            SFT_BATCH_SIZE,
            CONTEXT_LENGTH,
            rng
        )

        with torch.autocast(
            device_type=device.type,
            dtype=AMP_DTYPE,
            enabled=USE_AMP
        ):

            _, loss = model(xb, yb)

        total_loss += loss.item()

    avg_loss = total_loss / EVAL_BATCHES

    model.train(was_training)

    return avg_loss


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(path, model, optimizer, step, val_loss):

    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "sft_config": {
                "learning_rate": SFT_LR,
                "batch_size": SFT_BATCH_SIZE,
                "steps": SFT_STEPS,
                "seed": SEED,
                "prompt": SFT_PROMPT,
            },
        },
        path
    )


# ============================================================
# MAIN TRAINING
# ============================================================

if __name__ == "__main__":

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"Using device: {device}")

    print("\nLoading SFT dataset...")

    train_split, val_split = load_sft_dataset()

    # --------------------------------------------------------
    # Load pretrained model
    # --------------------------------------------------------

    model = ModernLLM(
        vocab_size,
        DIM,
        N_HEADS,
        N_LAYERS,
        CONTEXT_LENGTH
    ).to(device)

    if not os.path.exists(BASE_CHECKPOINT):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {BASE_CHECKPOINT}"
        )

    ckpt = torch.load(
        BASE_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True
    )

    print(
        f"\nLoaded pretrained model "
        f"(step {ckpt['step']}, "
        f"Val loss: {ckpt['val_loss']:.4f})"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SFT_LR,
        weight_decay=0.01
    )

    train_rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1)

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    initial_val_loss = evaluate_sft(
        model,
        val_split,
        val_rng
    )

    print(
        f"\nInitial SFT validation loss: "
        f"{initial_val_loss:.4f}"
    )

    best_val_loss = initial_val_loss

    save_checkpoint(
        BEST_SFT_CHECKPOINT,
        model,
        optimizer,
        0,
        best_val_loss
    )

    # --------------------------------------------------------
    # SFT training loop
    # --------------------------------------------------------

    print("\nStarting SFT training...\n")

    model.train()

    for step in range(1, SFT_STEPS + 1):

        xb, yb = build_sft_batch(
            train_split,
            SFT_BATCH_SIZE,
            CONTEXT_LENGTH,
            train_rng
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=AMP_DTYPE,
            enabled=USE_AMP
        ):

            _, loss = model(xb, yb)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at step {step}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # Periodic validation
        # ----------------------------------------------------

        if step % EVAL_INTERVAL == 0 or step == SFT_STEPS:

            val_loss = evaluate_sft(
                model,
                val_split,
                val_rng
            )

            print(
                f"SFT Step {step:4d}/{SFT_STEPS} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                save_checkpoint(
                    BEST_SFT_CHECKPOINT,
                    model,
                    optimizer,
                    step,
                    best_val_loss
                )

                print(
                    f"  New best checkpoint saved! "
                    f"Val Loss: {best_val_loss:.4f}"
                )

    # --------------------------------------------------------
    # Save final checkpoint
    # --------------------------------------------------------

    save_checkpoint(
        LAST_SFT_CHECKPOINT,
        model,
        optimizer,
        SFT_STEPS,
        val_loss
    )

    print("\nSFT training completed.")

    print(f"Best validation loss: {best_val_loss:.4f}")

    print(f"Best checkpoint: {BEST_SFT_CHECKPOINT}")

    print(f"Final checkpoint: {LAST_SFT_CHECKPOINT}")

    # --------------------------------------------------------
    # Load best checkpoint for generation
    # --------------------------------------------------------

    best_ckpt = torch.load(
        BEST_SFT_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        best_ckpt["model_state_dict"],
        strict=True
    )

    model.eval()

    # --------------------------------------------------------
    # Sample generation
    # --------------------------------------------------------

    test_prompt = SFT_PROMPT

    prompt_tensor = torch.tensor(
        [encode(test_prompt)],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        out = model.generate(
            prompt_tensor,
            max_new_tokens=220,
            use_cache=True
        )[0].tolist()

    if EOT_TOKEN in out:
        out = out[:out.index(EOT_TOKEN)]

    print("\n--- BEST SFT SAMPLE OUTPUT ---\n")

    print(decode(out))

    print("\nSFT pipeline finished successfully.")

