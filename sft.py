import os
import random
import re
import torch
from datasets import load_dataset

from config import CONTEXT_LENGTH, DIM, N_HEADS, N_LAYERS, device
from data import EOT_TOKEN, decode, encode, vocab_size
from model import ModernLLM

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

SFT_STEPS = 800
SFT_LR = 3e-5
SFT_BATCH_SIZE = 16

EVAL_INTERVAL = 50
EVAL_BATCHES = 10

UNCOND_PROMPT = "Instruction: Write a short story.\nStory:\n"
CFG_DROPOUT_PROB = 0.15  # 15% unconditional prompt dropout to enable CFG at inference

CHECKPOINT_DIR = "checkpoints"
BASE_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")
BEST_SFT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sft_best.pt")
LAST_SFT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "sft_last.pt")

USE_AMP = device.type in {"mps", "cuda"}
AMP_DTYPE = torch.bfloat16

# ============================================================
# PRECISE PROTAGONIST & PLOT EXTRACTION
# ============================================================

SUBJECT_MAP = {
    "robot": "a little robot",
    "dragon": "a friendly dragon",
    "monster": "a friendly monster",
    "princess": "a kind princess",
    "wizard": "a clever wizard",
    "fairy": "a magical fairy",
    "dog": "a playful dog",
    "puppy": "a playful puppy",
    "cat": "a curious cat",
    "kitten": "a curious kitten",
    "rabbit": "a brave rabbit",
    "bunny": "a brave bunny",
    "bear": "a cheerful bear",
    "bird": "a little bird",
    "owl": "a wise little owl",
    "duck": "a happy little duck",
    "frog": "a green little frog",
    "turtle": "a slow little turtle",
    "mouse": "a tiny mouse",
    "squirrel": "a busy squirrel",
    "monkey": "a funny monkey",
    "lion": "a brave lion",
    "elephant": "a gentle elephant",
    "boy": "a little boy",
    "girl": "a little girl",
}

# Matches the first character introduced in sentence 1 (prevents toy/secondary noun confusion)
PROTAGONIST_RE = re.compile(
    r"\b(?:there was|lived|named|called|about)\b[^.!?]{0,55}?\b("
    + "|".join(SUBJECT_MAP.keys())
    + r")\b",
    re.IGNORECASE,
)

NAME_RE = re.compile(
    r"\b(?:named|called)\s+([A-Z][a-z]{2,10})\b"
)

ACTIVITY_RULES = [
    (
        re.compile(r"\b(cook|cooked|cooking|bake|baked|baking|chef|made soup|make soup|made a cake|make a cake)\b", re.I),
        "who wants to cook and make yummy food",
    ),
    (
        re.compile(r"\b(went to the park|at the park|in the park|playground|down the slide|on the swings|adventure)\b", re.I),
        "who goes on an adventure at the park",
    ),
    (
        re.compile(r"\b(learned to fly|flew high|fly in the sky|flapped (?:his|her|its) wings)\b", re.I),
        "who wants to fly high in the sky",
    ),
    (
        re.compile(r"\b(magic|magical|wand|cast a spell|secret wish)\b", re.I),
        "who discovers something magical",
    ),
    (
        re.compile(r"\b(toy car|toy truck|toy train|red car|blue truck)\b", re.I),
        "who loves playing with a favorite toy vehicle",
    ),
    (
        re.compile(r"\b(painted a picture|drew a picture|sang a song|loved to paint)\b", re.I),
        "who loves making art and music",
    ),
    (
        re.compile(r"\b(was brave|very brave|not afraid anymore|helped save|rescued)\b", re.I),
        "who learns how to be brave",
    ),
    (
        re.compile(r"\b(shared (?:his|her)|helped (?:his|her) friend|kind to others)\b", re.I),
        "who learns the importance of helping others",
    ),
]

# High-precision compositional examples to reinforce multi-constraint adherence
COMPOSITIONAL_ANCHORS = [
    (
        "Instruction: Write a story about a little robot named Max who wants to cook and make yummy food.\nStory:\n",
        "Once upon a time, there was a little robot named Max. Max had shiny metal hands and loved working in the kitchen because he wanted to be a chef and cook yummy food.\n\nOne sunny morning, Max decided to bake a warm apple pie and make sweet soup for his friends. He carefully mixed the flour, sliced the red apples, and put the pan into the oven.\n\nSoon, a delicious smell filled the whole kitchen! His friends came over and tasted the warm pie. Everyone cheered for Max the robot chef, and Max beeped happily with pride.",
    ),
    (
        "Instruction: Write a story about a little robot named Leo who wants to cook and make yummy food.\nStory:\n",
        "Once upon a time, there was a little robot named Leo who dreamed of cooking delicious meals in a big kitchen. Every day, Leo put on his white chef hat and practiced stirring warm soup in a big pot.\n\nOne day, Leo baked soft chocolate cookies and cooked fresh pasta for his family. At first, he spilled a little flour on his metal feet, but he cleaned it up and kept trying.\n\nWhen the cookies came out of the oven, they were warm and sweet. Everyone thanked Leo for cooking such yummy food, and Leo was the happiest little robot chef in town.",
    ),
    (
        "Instruction: Write a story about a brave rabbit named Benny who goes on an adventure at the park.\nStory:\n",
        "Once upon a time, there was a brave little rabbit named Benny with long soft ears. One sunny morning, Benny hopped out of his burrow to go on a big adventure at the park.\n\nAt the park, Benny climbed up the tall slide and explored the green bushes near the pond. Suddenly, he saw a tiny kitten stuck behind a big log near the swings.\n\nBenny was very brave. He hopped over the log, showed the kitten the safe path back to the playground, and shared his crunchy carrot. Benny and the kitten played together at the park all afternoon.",
    ),
    (
        "Instruction: Write a story about a brave rabbit named Toby who goes on an adventure at the park.\nStory:\n",
        "Once upon a time, there was a brave rabbit named Toby who loved exploring new places. One afternoon, Toby went on an exciting adventure at the sunny park.\n\nHe hopped past the swings, crossed the little wooden bridge, and climbed to the top of the grassy hill. Along the way, a strong wind blew a little bird's ball into the bushes.\n\nToby bravely hopped into the thick bushes and pushed the ball back out with his nose. The little bird chirped with joy, and Toby was proud of his wonderful park adventure.",
    ),
    (
        "Instruction: Write a story about a playful dog named Max who goes on an adventure at the park.\nStory:\n",
        "Once upon a time, there was a playful dog named Max who loved running outside. One bright morning, Max went on a fun adventure at the park with his favorite red ball.\n\nMax ran around the big oak trees, jumped over a small puddle, and met a friendly puppy near the slide. They chased the red ball across the green grass together.\n\nWhen the sun began to set, Max wagged his tail happily and walked home after the best park adventure ever.",
    ),
]


def extract_dynamic_instruction(story_text):
    first_sentence = re.split(r"[.!?]", story_text, maxsplit=1)[0]

    subj_match = PROTAGONIST_RE.search(first_sentence)
    if not subj_match:
        return None
    raw_subj = subj_match.group(1).lower()
    matched_subject = SUBJECT_MAP[raw_subj]

    name_match = NAME_RE.search(story_text[:180])
    matched_name = name_match.group(1) if name_match else None

    matched_activity = None
    for pattern, desc in ACTIVITY_RULES:
        if pattern.search(story_text[:500]):
            matched_activity = desc
            break

    if matched_name is None and matched_activity is None:
        return None

    parts = [f"Write a story about {matched_subject}"]
    if matched_name:
        parts.append(f"named {matched_name}")
    if matched_activity:
        parts.append(matched_activity)

    return f"Instruction: {' '.join(parts)}.\nStory:\n"


# ============================================================
# REPRODUCIBILITY & DATASET PREPARATION
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_sft_dataset():
    dataset = load_dataset("roneneldan/TinyStories")
    if "validation" in dataset:
        train_split = dataset["train"]
        val_split = dataset["validation"]
    else:
        split = dataset["train"].train_test_split(test_size=0.01, seed=SEED)
        train_split = split["train"]
        val_split = split["test"]

    print(f"Training examples: {len(train_split):,}")
    print(f"Validation examples: {len(val_split):,}")
    return train_split, val_split


# ============================================================
# BATCH CONSTRUCTION
# ============================================================

def build_sft_batch(dataset_split, batch_size, context_length, rng, is_training=True):
    input_batch = []
    target_batch = []

    attempts = 0
    max_attempts = batch_size * 150

    while len(input_batch) < batch_size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Unable to construct a complete SFT batch.")

        # Mix in clean compositional anchor examples during training (20% of batch items)
        if is_training and rng.random() < 0.20:
            prompt_str, story = rng.choice(COMPOSITIONAL_ANCHORS)
        else:
            idx = rng.randrange(len(dataset_split))
            story = dataset_split[idx]["text"].strip()
            if not story:
                continue
            prompt_str = extract_dynamic_instruction(story)
            if prompt_str is None:
                continue

        # Apply 15% unconditional prompt dropout during training so CFG works at inference
        if is_training and rng.random() < CFG_DROPOUT_PROB:
            prompt_str = UNCOND_PROMPT

        prompt_ids = encode(prompt_str)
        story_ids = encode(story)

        available_story_tokens = context_length - len(prompt_ids)
        if len(story_ids) < 25 or len(story_ids) > available_story_tokens:
            continue

        full_ids = prompt_ids + story_ids + [EOT_TOKEN]

        x = full_ids[:-1]
        y = full_ids[1:]

        prompt_cutoff = len(prompt_ids) - 1
        y_masked = [-100 if i < prompt_cutoff else token for i, token in enumerate(y)]

        pad_len = context_length - len(x)
        if pad_len > 0:
            x = x + [EOT_TOKEN] * pad_len
            y_masked = y_masked + [-100] * pad_len

        input_batch.append(torch.tensor(x, dtype=torch.long))
        target_batch.append(torch.tensor(y_masked, dtype=torch.long))

    return torch.stack(input_batch).to(device), torch.stack(target_batch).to(device)


# ============================================================
# VALIDATION & CHECKPOINT SAVING
# ============================================================

@torch.no_grad()
def evaluate_sft(model, val_split):
    was_training = model.training
    model.eval()
    eval_rng = random.Random(SEED + 100)

    total_loss = 0.0
    for _ in range(EVAL_BATCHES):
        xb, yb = build_sft_batch(
            val_split, SFT_BATCH_SIZE, CONTEXT_LENGTH, eval_rng, is_training=False
        )
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            _, loss = model(xb, yb)
        total_loss += loss.item()

    model.train(was_training)
    return total_loss / EVAL_BATCHES


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
                "dynamic_prompts": True,
                "cfg_dropout": CFG_DROPOUT_PROB,
            },
        },
        path,
    )


# ============================================================
# MAIN TRAINING
# ============================================================

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Using device: {device}\nLoading SFT dataset...")
    train_split, val_split = load_sft_dataset()

    model = ModernLLM(vocab_size, DIM, N_HEADS, N_LAYERS, CONTEXT_LENGTH).to(device)
    if not os.path.exists(BASE_CHECKPOINT):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {BASE_CHECKPOINT}")

    ckpt = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"\nLoaded pretrained model (step {ckpt['step']}, Val loss: {ckpt['val_loss']:.4f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)
    train_rng = random.Random(SEED)

    initial_val_loss = evaluate_sft(model, val_split)
    print(f"Initial SFT validation loss: {initial_val_loss:.4f}\n")
    best_val_loss = initial_val_loss
    save_checkpoint(BEST_SFT_CHECKPOINT, model, optimizer, 0, best_val_loss)

    model.train()
    for step in range(1, SFT_STEPS + 1):
        xb, yb = build_sft_batch(
            train_split, SFT_BATCH_SIZE, CONTEXT_LENGTH, train_rng, is_training=True
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            _, loss = model(xb, yb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == SFT_STEPS:
            val_loss = evaluate_sft(model, val_split)
            print(f"SFT Step {step:4d}/{SFT_STEPS} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(BEST_SFT_CHECKPOINT, model, optimizer, step, best_val_loss)
                print(f"  New best checkpoint saved! Val Loss: {best_val_loss:.4f}")

    save_checkpoint(LAST_SFT_CHECKPOINT, model, optimizer, SFT_STEPS, val_loss)
    print(f"\nSFT training completed. Best validation loss: {best_val_loss:.4f}")

    # Load best SFT checkpoint and generate with Classifier-Free Guidance + Sentence Completion
    best_ckpt = torch.load(BEST_SFT_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    model.eval()

    test_prompts = [
        "Instruction: Write a story about a little robot named Max who wants to cook and make yummy food.\nStory:\n",
        "Instruction: Write a story about a brave rabbit named Benny who goes on an adventure at the park.\nStory:\n",
    ]

    uncond_ids = encode(UNCOND_PROMPT)
    for p in test_prompts:
        prompt_tensor = torch.tensor([encode(p)], dtype=torch.long, device=device)
        uncond_tensor = torch.tensor([uncond_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                prompt_tensor,
                max_new_tokens=220,
                use_cache=True,
                cfg_scale=1.75,
                uncond_idx=uncond_tensor,
                grace_tokens=50,
            )[0].tolist()
        if EOT_TOKEN in out:
            out = out[: out.index(EOT_TOKEN)]
        print("\n--- SAMPLE OUTPUT ---\n" + decode(out).strip())
