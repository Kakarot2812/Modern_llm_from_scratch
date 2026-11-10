# Modern LLM from Scratch in PyTorch

A lightweight, decoder-only Transformer language model built from scratch in PyTorch. Unlike classic GPT-2 implementations that rely on learned absolute positional embeddings, standard LayerNorm, and GELU activations, **ModernLLM** incorporates modern architectural components used in LLaMA and Mistral models—including **RMSNorm**, **Rotary Positional Embeddings (RoPE)**, and **SwiGLU** feed-forward networks—trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

---

## Features & Architecture

* **RMSNorm (Root Mean Square Normalization):** Pre-normalization applied before attention and feed-forward blocks for faster, more stable training:
  $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$
* **RoPE (Rotary Positional Embeddings):** Encodes relative token positions by rotating query and key representations by angle multiples in 2D subspaces rather than adding fixed positional vectors.
* **SwiGLU Feed-Forward Network:** Gated linear unit with SiLU ($\text{Swish}$) activation and bias-free linear projections:
  $$\text{FFN}(x) = W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big)$$
* **Causal Multi-Head Self-Attention:** Bias-free query, key, value, and output projections with causal masking.
* **GPT-2 BPE Tokenization:** Uses `tiktoken` (`gpt2` encoding with vocabulary size $V = 50,257$).
* **Hardware Acceleration:** Automatically detects and runs on Apple Silicon (`mps`), NVIDIA GPUs (`cuda`), or `cpu`.
* **Fault-Tolerant Checkpointing:** Atomic checkpoint saves (`latest.pt`, `best.pt`, and rolling step snapshots) allow training to be interrupted and resumed without data corruption or lost loss history.

---

## Project Structure

```text
modern_llm/
├── config.py      # Device detection and model/training hyperparameters
├── data.py        # TinyStories dataset loader, GPT-2 BPE tokenizer, and batch sampler
├── layers.py      # Custom RMSNorm module and RoPE frequency/rotation functions
├── model.py       # CausalSelfAttention, SwiGLU FeedForward, Transformer Block, and ModernLLM
├── train.py       # Training loop with evaluation, atomic checkpointing, and loss plotting
├── generate.py    # Standalone inference script with Temperature, Top-K, Top-P, and Repetition Penalty
└── README.md      # Project documentation
```

> **Note on Local Artifacts:** Model weights (`checkpoints/`) and training plots (`loss.png`) are generated locally during training and are excluded from version control.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd modern_llm
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch datasets tiktoken matplotlib
   ```

---

## Usage

### 1. Train the Model
Run the training script:
```bash
python train.py
```
* On startup, `data.py` downloads and tokenizes the `roneneldan/TinyStories` dataset ($\approx 474\text{M}$ training tokens and $\approx 4.77\text{M}$ validation tokens).
* Checkpoints are automatically saved to `checkpoints/latest.pt` and `checkpoints/best.pt` every `EVAL_INTERVAL` steps.
* **Resuming Training:** If `checkpoints/latest.pt` exists, running `python train.py` automatically resumes from the last completed step and preserves your loss history.
* Upon completion, a training and validation loss curve is saved locally to `loss.png`.

### 2. Generate Text
Once a checkpoint is saved in `checkpoints/best.pt`, generate stories using:
```bash
python generate.py
```
You can customize the prompt and decoding settings at the top of `generate.py`:
* `PROMPT`: Starting text for story generation.
* `MAX_NEW_TOKENS`: Target number of tokens to generate before completing the final sentence.
* `TEMPERATURE`: Controls randomness (e.g., $0.65$ for focused stories, $0.85$ for more creative outputs).
* `TOP_K` & `TOP_P`: Top-$k$ and nucleus ($p$) sampling filters to prevent low-probability token drift.
* `REPETITION_PENALTY`: Penalizes tokens in a local 16-token window to prevent phrase loops while keeping main character names active.
* `INTERACTIVE`: Set to `True` to enter an interactive terminal prompt loop without reloading the checkpoint.

---

## Model Configuration

Default hyperparameters defined in `config.py`:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `DIM` | $256$ | Embedding and hidden state dimension |
| `N_LAYERS` | $6$ | Number of Transformer decoder blocks |
| `N_HEADS` | $8$ | Number of attention heads (head dimension $= 32$) |
| `CONTEXT_LENGTH` | $128$ | Maximum sequence length |
| `BATCH_SIZE` | $16$ | Training batch size |
| `DROPOUT` | $0.2$ | Dropout probability |
| `LR` | $3 \times 10^{-4}$ | Learning rate for AdamW optimizer |
| `MAX_ITERS` | $6000$ | Total training iterations |
| `EVAL_INTERVAL` | $250$ | Steps between validation evaluations and checkpoints |

---

## Training Results

Trained on Apple Silicon (`mps`) for $6,000$ steps:

| Step | Train Loss | Validation Loss | Checkpoint Status |
| :---: | :---: | :---: | :--- |
| $0$ | $10.7740$ | $10.7803$ | Initial baseline |
| $500$ | $3.4085$ | $3.3900$ | Saved `best.pt` |
| $1000$ | $2.9739$ | $2.9796$ | Saved `best.pt` |
| $2000$ | $2.6475$ | $2.6342$ | Saved `best.pt` |
| $3000$ | $2.4482$ | $2.4684$ | Saved `best.pt` |
| $4000$ | $2.3677$ | $2.3778$ | Saved `best.pt` |
| $5000$ | $2.2844$ | $2.2959$ | Saved `best.pt` |
| $5750$ | $2.2024$ | **$2.2269$** | **Best validation loss (`best.pt`)** |

### Sample Generation (`best.pt` at Step $5750$, Val Loss: $2.2269$)

**Prompt:**
```text
Once upon a time, there was a little dog named Max who loved to
```

**Output:**
```text
Once upon a time, there was a little dog named Max who loved to play. One day, Max's friend Billy came over to play. They played games and had lots of fun together.

After playing with their friends, they decided to go for a walk in the park...
```

---