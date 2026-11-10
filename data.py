import os
from datasets import load_dataset
import torch
import tiktoken
from config import device

# --- Download and read data ---
dataset = load_dataset("roneneldan/TinyStories")

train_text = "\n".join(dataset["train"]["text"])
val_text = "\n".join(dataset["validation"]["text"])

print("Train characters:", len(train_text))
print("Validation characters:", len(val_text))


# --- byte Pair Encoding (BPE) replacing character vocabulary ---
bpe_tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = bpe_tokenizer.n_vocab

# Encode/decode helpers (same signatures as notebook)
def encode(s):
    return bpe_tokenizer.encode(s)

def decode(ids):
    return bpe_tokenizer.decode(ids)

# --- Tokenize and split ---
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

print("Train tokens:", len(train_data))
print("Validation tokens:", len(val_data))

# ---  Batching: grab random chunks of text ---
def get_batch(split, batch_size, context_length):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length, (batch_size,))
    x = torch.stack([d[i:i+context_length] for i in ix])
    y = torch.stack([d[i+1:i+context_length+1] for i in ix])
    return x.to(device), y.to(device)
