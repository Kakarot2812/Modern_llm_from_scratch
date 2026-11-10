import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DROPOUT, device
from layers import RMSNorm, precompute_rope_freqs, apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        wei = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len)
        self.ln2 = RMSNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


class ModernLLM(nn.Module):
    def __init__(self, vocab_size, dim, n_heads, n_layers, context_length):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, context_length) for _ in range(n_layers)])
        self.ln_f = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        cos, sin = precompute_rope_freqs(dim // n_heads, context_length)
        self.cos = cos.to(device)
        self.sin = sin.to(device)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
