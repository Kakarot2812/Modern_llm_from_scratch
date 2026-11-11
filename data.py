# import os
# from datasets import load_dataset
# import torch
# import tiktoken
# from config import device

# # --- Download and read data ---
# dataset = load_dataset("roneneldan/TinyStories")

# train_text = "\n".join(dataset["train"]["text"])
# val_text = "\n".join(dataset["validation"]["text"])

# print("Train characters:", len(train_text))
# print("Validation characters:", len(val_text))


# # --- byte Pair Encoding (BPE) replacing character vocabulary ---
# bpe_tokenizer = tiktoken.get_encoding("gpt2")
# vocab_size = bpe_tokenizer.n_vocab

# # Encode/decode helpers (same signatures as notebook)
# def encode(s):
#     return bpe_tokenizer.encode(s)

# def decode(ids):
#     return bpe_tokenizer.decode(ids)

# # --- Tokenize and split ---
# train_data = torch.tensor(encode(train_text), dtype=torch.long)
# val_data = torch.tensor(encode(val_text), dtype=torch.long)

# print("Train tokens:", len(train_data))
# print("Validation tokens:", len(val_data))

# # ---  Batching: grab random chunks of text ---
# def get_batch(split, batch_size, context_length):
#     d = train_data if split == "train" else val_data
#     ix = torch.randint(len(d) - context_length, (batch_size,))
#     x = torch.stack([d[i:i+context_length] for i in ix])
#     y = torch.stack([d[i+1:i+context_length+1] for i in ix])
#     return x.to(device), y.to(device)

import os
from datasets import load_dataset
import torch
import tiktoken
from config import device

bpe_tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = bpe_tokenizer.n_vocab
EOT_TOKEN = bpe_tokenizer.eot_token  # 50256 (<|endoftext|>)


def encode(s):
    return bpe_tokenizer.encode(s, allowed_special={"<|endoftext|>"})


def decode(ids):
    return bpe_tokenizer.decode(ids)


CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
train_cache = os.path.join(CACHE_DIR, "train_eot_tokens.pt")
val_cache = os.path.join(CACHE_DIR, "val_eot_tokens.pt")

if os.path.exists(train_cache) and os.path.exists(val_cache):
    print("Loading cached tokenized dataset with <|endoftext|> boundaries...")
    train_data = torch.load(train_cache, weights_only=True)
    val_data = torch.load(val_cache, weights_only=True)
else:
    print("Tokenizing dataset with <|endoftext|> story boundaries (one-time cache)...")
    dataset = load_dataset("roneneldan/TinyStories")

    eot_str = "<|endoftext|>\n"
    train_text = eot_str.join(dataset["train"]["text"]) + eot_str
    val_text = eot_str.join(dataset["validation"]["text"]) + eot_str

    train_data = torch.tensor(encode(train_text), dtype=torch.long)
    val_data = torch.tensor(encode(val_text), dtype=torch.long)

    torch.save(train_data, train_cache)
    torch.save(val_data, val_cache)

print(f"Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")


def get_batch(split, batch_size, context_length):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length, (batch_size,))
    x = torch.stack([d[i : i + context_length] for i in ix])
    y = torch.stack([d[i + 1 : i + context_length + 1] for i in ix])
    return x.to(device), y.to(device)
