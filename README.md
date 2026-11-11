# ModernLLM — A Modern Language Model Built from Scratch

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Kakarot2812/modern-llm-from-scratch)

A compact causal language model implemented in **PyTorch**, featuring modern Transformer components, controlled architectural ablations, autoregressive KV-cache inference, and supervised fine-tuning.

ModernLLM explores how modern LLM architecture choices affect model quality, parameter efficiency, training behavior, and inference performance on consumer hardware.

The project replaces traditional GPT-2-style components with:

- Rotary Position Embeddings (RoPE)
- Root Mean Square Normalization (RMSNorm)
- SwiGLU feed-forward networks
- Tied token embeddings and output projection weights
- Bias-free Multi-Head Attention with fused `F.scaled_dot_product_attention` (SDPA)
- Autoregressive KV caching and Classifier-Free Guidance (CFG)

The model was pretrained on **TinyStories** and developed on Apple Silicon using PyTorch MPS.

**Live Interactive Demo:** [Hugging Face Spaces — ModernLLM](https://huggingface.co/spaces/Kakarot2812/modern-llm-from-scratch)

---

## Key Results

| Experiment | Result |
|---|---:|
| ModernLLM parameters (tied) | $19.16\text{M}$ |
| GPT-2 baseline parameters | $30.53\text{M}$ |
| Pretraining steps | $6,000$ |
| Tokens processed during pretraining | $\sim 49.15\text{M}$ |
| Best pretraining validation loss | $1.8875$ |
| Pretraining perplexity | $6.60$ |
| Best SFT response validation loss | $1.7402$ |
| Controlled 50-prompt rubric score (Base vs. SFT) | $59.37 \rightarrow 75.28$ ($+15.91\text{ pts}$) |
| KV-cache inference speed | $292.49\text{ tokens/sec}$ |
| Measured KV-cache speedup | $2.04\times$ |

**Highlights**

- Approximately $37.2\%$ fewer parameters than the GPT-2 baseline.
- Approximately $21.5\%$ lower perplexity than the GPT-2 baseline in the reported ablation experiment.
- Approximately $40.2\%$ fewer parameters compared with the untied ModernLLM configuration.
- A custom autoregressive KV-cache implementation achieving a measured $2.04\times$ speedup in the reported benchmark.
- A $+15.91\text{-point}$ ($+26.8\%$ relative) improvement on a controlled $50$-prompt story generation benchmark from supervised fine-tuning alone.

*All performance measurements are experimental results from the configurations described below. Inference speed depends on hardware, sequence length, and generation settings.*

---

## Architecture

| Component | GPT-2 baseline | ModernLLM |
|---|---|---|
| Positional encoding | Learned absolute embeddings | RoPE |
| Normalization | LayerNorm | RMSNorm |
| Feed-forward network | GELU | SwiGLU |
| Attention | Manual causal masked attention | Fused SDPA (`scaled_dot_product_attention`) |
| Linear projections | Bias enabled | Bias-free |
| Weight sharing | Untied | Tied |
| Inference | Standard autoregressive | KV-cache enabled + optional CFG |

### Architecture details

**1. Rotary Position Embeddings (RoPE)**

RoPE encodes positional information through rotations applied to query and key representations at each layer, supporting dynamic `start_pos` offsets during cached decoding.

**2. RMSNorm**

Uses root mean square normalization instead of mean-and-variance LayerNorm for pre-attention, pre-FFN, and final layer normalization.

**3. SwiGLU**

Uses a gated feed-forward architecture with the SiLU activation function and three bias-free linear projections.

**4. Fused Scaled Dot-Product Attention (SDPA)**

Uses PyTorch's fused `F.scaled_dot_product_attention` with bias-free query, key, value, and output projections, applying causal masking during prefill/training and single-token unmasked attention over cached keys during autoregressive decoding.

**5. Weight tying**

Shares the token embedding matrix with the language-model output projection, reducing the number of trainable parameters by $12.87\text{M}$.

**6. KV Cache**

Stores previously computed key and value tensors during autoregressive generation, avoiding redundant computation for earlier tokens.

---

## Controlled Architecture Ablation

Three configurations were evaluated using the same reported training setup:

- Batch size: $16$
- Sequence length: $256$
- Optimizer: AdamW
- Learning rate: $5 \times 10^{-4}$
- Training steps: $1,000$

### Results

| Architecture | Parameters | Validation Loss | Perplexity | Training Time |
|---|---:|---:|---:|---:|
| GPT-2 baseline (LearnedPos + LayerNorm + GELU) | $30.53\text{M}$ | $3.0835$ | $21.83$ | $462.5\text{s}$ |
| ModernLLM untied (RoPE + RMSNorm + SwiGLU) | $32.03\text{M}$ | $2.7809$ | $16.13$ | $583.1\text{s}$ |
| ModernLLM tied | $19.16\text{M}$ | $2.8413$ | $17.14$ | $596.8\text{s}$ |

### Observations

1. Replacing the legacy architecture components with RoPE, RMSNorm, and SwiGLU reduced perplexity from $21.83$ to $16.13$ in this experiment.
2. Weight tying reduced the parameter count from $32.03\text{M}$ to $19.16\text{M}$.
3. The tied configuration retained much of the improvement over the baseline while using substantially fewer parameters.
4. The tied configuration had slightly higher validation loss than the untied configuration.

These results demonstrate the observed trade-offs in these particular experiments; they do not isolate the contribution of every individual architectural component.

![Architecture ablation results](ablation_results.png)

---

## Pretraining

ModernLLM was pretrained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) using `<|endoftext|>` document boundaries.

### Training configuration

| Setting | Value |
|---|---|
| Architecture | Tied ModernLLM |
| Parameters | $19.16\text{M}$ |
| Layers / Heads / Dim | $6$ / $8$ / $256$ |
| Context length | $512$ |
| Batch size | $16$ |
| Training steps | $6,000$ |
| Optimizer | AdamW |
| Peak learning rate | $5 \times 10^{-4}$ |
| Weight decay | $0.1$ |
| Warmup | $300\text{ steps}$ |
| Scheduler | Cosine decay |
| Final learning rate | $5 \times 10^{-5}$ |
| Gradient clipping | $1.0$ |

### Training trajectory

| Step | Train Loss | Validation Loss | Perplexity |
|---:|---:|---:|---:|
| $0$ | $10.8611$ | $10.8615$ | $52,131.10$ |
| $1,000$ | $2.6749$ | $2.6831$ | $14.63$ |
| $3,000$ | $2.1293$ | $2.1055$ | $8.21$ |
| $4,500$ | $1.9621$ | $1.9331$ | $6.91$ |
| $5,999$ | $1.8825$ | $1.8875$ | $6.60$ |

The final reported pretraining checkpoint is:

`checkpoints/best.pt`

The model processed approximately $49.15\text{ million}$ training tokens over $6,000$ steps.

![Pretraining loss curve](loss.png)

---

## Evaluation

The evaluation pipeline (`evaluate.py`) measures language-model loss, generation diversity, repetition, entity retention, completion behavior, and inference throughput on the pretrained model.

### Reported evaluation metrics

| Metric | Result |
|---|---:|
| Validation loss | $1.9291$ |
| Validation perplexity | $6.88$ |
| Distinct-1 | $0.3166$ |
| Distinct-2 | $0.7770$ |
| Distinct-3 | $0.9392$ |
| 4-gram repetition rate | $0.0065$ |
| Entity retention rate | $75\%$ |
| Clean completion rate | $100\%$ |
| KV cache disabled | $143.69\text{ tok/s}$ |
| KV cache enabled | $292.49\text{ tok/s}$ |
| Speedup | $2.04\times$ |

### Metric interpretation

- **Distinct-$n$:** Measures lexical diversity across generated text.
- **4-gram repetition rate:** Measures repeated four-token sequences.
- **Entity retention:** Measures whether introduced characters remain present in the second half of generated narratives.
- **Clean completion:** Measures whether outputs satisfy the implemented sentence-ending punctuation heuristic.
- **Inference throughput:** Measures generated tokens per second.

The clean completion rate is a heuristic-based metric, not a human assessment of narrative quality or coherence.

The evaluation report is saved to `eval_report.json`.

---

## Supervised Fine-Tuning (SFT)

After pretraining, the model was fine-tuned on TinyStories using dynamically extracted multi-slot instructions paired with full stories.

### Dynamic instruction format

For each candidate story, regex rules inspect the opening sentence to extract the protagonist species/role, character name, and plot activity:

```text
Instruction: Write a story about a brave rabbit named Benny who goes on an adventure at the park.
Story:
```

During training, $15\%$ of prompts are replaced with an unconditional fallback (`Instruction: Write a short story.\nStory:\n`) to support optional Classifier-Free Guidance (CFG) at inference time.

### Training configuration

| Setting | Value |
|---|---|
| Starting checkpoint | `checkpoints/best.pt` |
| Training steps | $800$ |
| Batch size | $16$ |
| Learning rate | $3 \times 10^{-5}$ |
| Optimizer | AdamW |
| Weight decay | $0.01$ |
| Unconditional prompt dropout | $15\%$ |
| Evaluation interval | $50\text{ steps}$ |
| Best checkpoint step | $800$ |

### Response-only loss masking

Prompt tokens and padding tokens are excluded from the cross-entropy training objective using `ignore_index = -100`. This ensures that gradients are computed strictly on the story response tokens rather than the instruction prefix.

### SFT results

| Metric | Result |
|---|---:|
| Initial SFT validation loss | $1.7997$ |
| Best SFT validation loss | $1.7402$ |
| Best checkpoint | Step $800$ |

The best checkpoint is saved as:

`checkpoints/sft_best.pt`

**Important scope limitation:** This experiment uses rule-extracted synthetic instructions and children's stories from TinyStories alongside five compositional anchor examples. It demonstrates instruction-conditioned story generation within the TinyStories domain, not general-purpose instruction following.

The SFT response validation loss is measured on a masked response-only objective over filtered story-instruction pairs and should not be compared directly against unmasked pretraining validation loss.

---

## Controlled Base vs. SFT Benchmark (50 Unseen Prompts)

A controlled comparison (`benchmark_eval.py`) was conducted across $50$ fixed unseen prompts spanning five balanced categories: *Animals*, *Children*, *Morals*, *Adventures*, and *Fantasy & Toys*.

To isolate the effect of supervised weight tuning from inference-time steering, both models received **identical formatted prompts** (`Instruction: <prompt>\nStory:\n`) and **identical decoding hyperparameters** with Classifier-Free Guidance disabled:

| Setting | Value |
|---|---|
| Prompts | $50$ |
| Maximum new tokens | $180$ ($+50$ grace tokens for sentence completion) |
| Temperature | $0.60$ |
| Top-$k$ | $30$ |
| Top-$p$ | $0.85$ |
| Repetition penalty | $1.18$ |
| Classifier-Free Guidance (CFG) | Disabled ($1.0$) for both models |

### Automated rubric results

| Dimension | Maximum | Base (`best.pt`) | SFT (`sft_best.pt`) | Delta |
|---|---:|---:|---:|---:|
| Character adherence | $20$ | $6.80$ | $15.04$ | $+8.24$ |
| Topic adherence | $20$ | $8.51$ | $10.68$ | $+2.17$ |
| Narrative coherence | $25$ | $20.90$ | $25.00$ | $+4.10$ |
| Character consistency | $15$ | $5.88$ | $6.96$ | $+1.08$ |
| Ending quality | $10$ | $8.24$ | $8.72$ | $+0.48$ |
| Repetition | $10$ | $9.04$ | $8.88$ | $-0.16$ |
| **Total** | **$100$** | **$59.37$** | **$75.28$** | **$+15.91$** |

The SFT model achieved a **$15.91$-point higher** average automated rubric score ($+26.8\%$ relative improvement) under strictly identical decoding conditions. When optional Classifier-Free Guidance ($\text{CFG} = 1.75$) is enabled at inference time for the SFT model, the composite automated score reaches **$79.17 / 100$**.

**Evaluation caveat:** These scores come from heuristic automated proxies, including early entity presence, keyword coverage, word and sentence counts, second-half entity retention, terminal punctuation, and $4$-gram repetition. They are not human-judged semantic quality scores or formal tests of statistical significance. A randomized, blinded A/B scoring sheet (`blinded_human_scoring_sheet.csv`) is generated alongside `benchmark_controlled_results.json` for manual human evaluation.

The experiment uses the same prompts and decoding hyperparameters, though identical random seeds do not guarantee identical token sampling trajectories across models with different output distributions.

---

## Repository Structure

```text
Modern_llm_from_scratch/
│
├── config.py                            # Hyperparameters and hardware device setup
├── data.py                              # TinyStories loading, BPE tokenization, and caching
├── layers.py                            # RMSNorm and RoPE implementations
├── model.py                             # ModernLLM architecture, KV cache, and CFG decoding
│
├── train.py                             # Pretraining pipeline with cosine LR schedule
├── ablation.py                          # Controlled architecture ablation experiments
├── evaluate.py                          # Pretrained model evaluation and KV-cache benchmark
├── sft.py                               # Dynamic instruction SFT pipeline with response masking
├── generate.py                          # Interactive and single-prompt text generation
├── compare.py                           # Side-by-side Base vs. SFT qualitative comparison
├── benchmark_prompts.py                 # 50 unseen prompts across 5 story categories
├── benchmark_eval.py                    # Controlled 50-prompt evaluation and blinded CSV export
├── export_weights.py                    # Exports standalone weights for deployment
│
├── ablation_results.png                 # Architecture ablation loss curves
├── loss.png                             # Pretraining train/validation loss curve
├── eval_report.json                     # Pretrained model evaluation metrics
├── benchmark_controlled_results.json    # Controlled 50-prompt benchmark report
├── blinded_human_scoring_sheet.csv      # Blinded A/B story pairs for human rubric grading
│
└── checkpoints/                         # Local training and SFT checkpoints (git-ignored)
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/Kakarot2812/Modern_llm_from_scratch.git
cd Modern_llm_from_scratch
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch datasets tiktoken matplotlib
```

### 4. Run architecture ablations

```bash
python ablation.py
```

Generates `ablation_results.png` and prints the architectural comparison table.

### 5. Pretrain ModernLLM

```bash
python train.py
```

Trains the model for $6,000$ steps and saves the best checkpoint to `checkpoints/best.pt` and the loss curve to `loss.png`.

### 6. Evaluate the pretrained model

```bash
python evaluate.py
```

Generates perplexity, diversity, entity retention, and KV-cache throughput metrics in `eval_report.json`.

### 7. Run supervised fine-tuning

```bash
python sft.py
```

Loads `checkpoints/best.pt`, fine-tunes on dynamically extracted instruction-story pairs for $800$ steps, and saves `checkpoints/sft_best.pt`.

### 8. Generate text

```bash
python generate.py
```

### 9. Compare pretrained and SFT models qualitatively

```bash
python compare.py
```

### 10. Run the controlled 50-prompt benchmark

```bash
python benchmark_eval.py
```

Evaluates both checkpoints under identical decoding settings and outputs `benchmark_controlled_results.json` and `blinded_human_scoring_sheet.csv`.

**Note:** Training and generation require the corresponding local checkpoints, or you can test the deployed SFT checkpoint directly on [Hugging Face Spaces](https://huggingface.co/spaces/Kakarot2812/modern-llm-from-scratch).

---

## Hardware and Implementation

Developed and tested on:

- Apple Silicon MacBook Air M4
- $16\text{ GB}$ unified memory
- macOS
- Python $3.12$
- PyTorch with MPS acceleration (`bfloat16` mixed precision)

The code automatically selects between MPS, CUDA, and CPU. Training time and inference throughput will vary across hardware configurations.

---

## Limitations and Future Work

Current limitations:

- **Domain specificity:** Training data is restricted to short children's stories from TinyStories; the model cannot follow general-purpose or non-narrative instructions.
- **Rule-based SFT extraction & anchor memorization:** SFT instructions are extracted via regex heuristics paired with five oversampled compositional anchors, which can cause memorization leakage (e.g., reverting to memorized cooking paragraphs) when prompted with out-of-distribution entities or rare character names.
- **Entity drift at $19.16\text{M}$ scale:** While initial character adherence is strong, secondary character names and pronouns can still drift over longer generations ($150+$ tokens).
- **Heuristic evaluation proxies:** Automated rubric metrics measure surface indicators (keyword presence, sentence counts, $n$-gram diversity) rather than deep semantic coherence.
- **Grouped architectural ablation:** The reported ablation study compares the full modern stack against the GPT-2 baseline rather than isolating RoPE, RMSNorm, and SwiGLU in separate single-variable runs.

Potential next improvements:

- Replace oversampled static compositional anchors with a diverse, synthetically generated or human-curated instruction-story dataset.
- Complete blinded human grading across all $50$ prompt pairs in `blinded_human_scoring_sheet.csv` to measure human win rates and correlation with the automated rubric.
- Compare multiple random seeds and single-component ablations for architectural experiments.
- Measure peak memory consumption during prefill and cached decoding across sequence lengths.

---

## References

- **TinyStories:** Eldan & Li, [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)
- **RoPE:** Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **RMSNorm:** Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- **SwiGLU:** Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- **Weight Tying:** Press & Wolf, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)

---

## Author

**Nitesh Yadav**

- GitHub: [@Kakarot2812](https://github.com/Kakarot2812)
- Repository: [Modern_llm_from_scratch](https://github.com/Kakarot2812/Modern_llm_from_scratch)
- Live Demo: [Hugging Face Spaces](https://huggingface.co/spaces/Kakarot2812/modern-llm-from-scratch)
