import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from config import DROPOUT, TIE_WEIGHTS
from layers import RMSNorm, precompute_rope_freqs, apply_rope

_tokenizer = tiktoken.get_encoding("gpt2")


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len, use_rope=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rope = use_rope

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(DROPOUT)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x, cos, sin, start_pos=0, kv_cache=None, use_cache=False):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, cos, sin, start_pos=start_pos)
            k = apply_rope(k, cos, sin, start_pos=start_pos)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        if kv_cache is not None and T == 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out)), new_kv_cache


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_type="swiglu"):
        super().__init__()
        self.ffn_type = ffn_type
        hidden_dim = 4 * dim
        if ffn_type == "swiglu":
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        else:
            self.c_fc = nn.Linear(dim, hidden_dim)
            self.c_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        if self.ffn_type == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len, use_rope=True, norm_type="rmsnorm", ffn_type="swiglu"):
        super().__init__()
        norm_cls = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.ln1 = norm_cls(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len, use_rope=use_rope)
        self.ln2 = norm_cls(dim)
        self.ffn = FeedForward(dim, ffn_type=ffn_type)

    def forward(self, x, cos, sin, start_pos=0, kv_cache=None, use_cache=False):
        attn_out, new_kv_cache = self.attn(
            self.ln1(x), cos, sin, start_pos=start_pos, kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv_cache


class ModernLLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        n_heads,
        n_layers,
        context_length,
        use_rope=True,
        norm_type="rmsnorm",
        ffn_type="swiglu",
        tie_weights=TIE_WEIGHTS,
    ):
        super().__init__()
        self.context_length = context_length
        self.use_rope = use_rope

        self.token_embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        self.pos_embedding = None if use_rope else nn.Embedding(context_length, dim)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [
                Block(dim, n_heads, context_length, use_rope=use_rope, norm_type=norm_type, ffn_type=ffn_type)
                for _ in range(n_layers)
            ]
        )
        norm_cls = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.ln_f = norm_cls(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        cos, sin = precompute_rope_freqs(dim // n_heads, context_length)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, targets=None, start_pos=0, kv_caches=None, use_cache=False):
        B, T = idx.shape
        x = self.token_embedding(idx)

        if not self.use_rope and self.pos_embedding is not None:
            positions = torch.arange(start_pos, start_pos + T, device=idx.device)
            x = x + self.pos_embedding(positions)

        new_kv_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(
                x, self.cos, self.sin, start_pos=start_pos, kv_cache=layer_cache, use_cache=use_cache
            )
            if use_cache:
                new_kv_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

        if use_cache:
            return logits, loss, new_kv_caches
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=220,
        temperature=0.65,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.12,
        use_cache=True,
        eot_token=50256,
        cfg_scale=1.0,
        uncond_idx=None,
        grace_tokens=50,
    ):
        kv_caches = None
        uncond_kv_caches = None
        total_Limit = max_new_tokens + (grace_tokens if eot_token is not None else 0)

        for step in range(total_Limit):
            if idx.size(1) > self.context_length:
                idx_cond = idx[:, -self.context_length :]
                kv_caches = None
            else:
                idx_cond = idx

            if use_cache and kv_caches is not None:
                start_pos = idx_cond.size(1) - 1
                logits, _, kv_caches = self(
                    idx_cond[:, -1:], start_pos=start_pos, kv_caches=kv_caches, use_cache=True
                )
            elif use_cache:
                logits, _, kv_caches = self(
                    idx_cond, start_pos=0, kv_caches=None, use_cache=True
                )
            else:
                logits, _ = self(idx_cond, start_pos=0, use_cache=False)

            logits = logits[:, -1, :].float()

            # Classifier-Free Guidance (CFG) to keep generation locked onto prompt entities & plot
            if cfg_scale > 1.0 and uncond_idx is not None:
                if uncond_idx.size(1) > self.context_length:
                    u_cond = uncond_idx[:, -self.context_length :]
                    uncond_kv_caches = None
                else:
                    u_cond = uncond_idx

                if use_cache and uncond_kv_caches is not None:
                    u_start = u_cond.size(1) - 1
                    u_logits, _, uncond_kv_caches = self(
                        u_cond[:, -1:], start_pos=u_start, kv_caches=uncond_kv_caches, use_cache=True
                    )
                elif use_cache:
                    u_logits, _, uncond_kv_caches = self(
                        u_cond, start_pos=0, kv_caches=None, use_cache=True
                    )
                else:
                    u_logits, _ = self(u_cond, start_pos=0, use_cache=False)

                u_logits = u_logits[:, -1, :].float()
                logits = u_logits + cfg_scale * (logits - u_logits)

            if repetition_penalty != 1.0:
                local_window = idx_cond[0, -24:]
                recent_tokens = torch.unique(local_window)
                recent_logits = logits[0, recent_tokens]
                logits[0, recent_tokens] = torch.where(
                    recent_logits < 0,
                    recent_logits * repetition_penalty,
                    recent_logits / repetition_penalty,
                )

            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eot_token is not None and idx_next.item() == eot_token:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            if uncond_idx is not None:
                uncond_idx = torch.cat((uncond_idx, idx_next), dim=1)

            # Once target length is reached, stop cleanly at the first sentence boundary
            if eot_token is not None and step >= max_new_tokens - 1:
                tail_text = _tokenizer.decode(idx[0, -12:].tolist()).rstrip()
                if tail_text.endswith((".", "!", "?", '."', '!"', '?"')):
                    break

        return idx


# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from config import DROPOUT, TIE_WEIGHTS
# from layers import RMSNorm, precompute_rope_freqs, apply_rope


# class CausalSelfAttention(nn.Module):
#     def __init__(self, dim, n_heads, max_seq_len, use_rope=True):
#         super().__init__()
#         self.n_heads = n_heads
#         self.head_dim = dim // n_heads
#         self.use_rope = use_rope

#         self.q = nn.Linear(dim, dim, bias=False)
#         self.k = nn.Linear(dim, dim, bias=False)
#         self.v = nn.Linear(dim, dim, bias=False)
#         self.proj = nn.Linear(dim, dim, bias=False)
#         self.dropout = nn.Dropout(DROPOUT)

#         self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

#     def forward(self, x, cos, sin, start_pos=0, kv_cache=None, use_cache=False):
#         B, T, C = x.shape
#         q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
#         k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
#         v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

#         if self.use_rope:
#             q = apply_rope(q, cos, sin, start_pos=start_pos)
#             k = apply_rope(k, cos, sin, start_pos=start_pos)

#         if kv_cache is not None:
#             k_prev, v_prev = kv_cache
#             k = torch.cat([k_prev, k], dim=2)
#             v = torch.cat([v_prev, v], dim=2)

#         new_kv_cache = (k, v) if use_cache else None

#         # During single-token cached decoding (T == 1), all cached keys are in the past
#         if kv_cache is not None and T == 1:
#             out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
#         else:
#             out = F.scaled_dot_product_attention(
#                 q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0
#             )

#         out = out.transpose(1, 2).contiguous().view(B, T, C)
#         return self.dropout(self.proj(out)), new_kv_cache


# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_type="swiglu"):
#         super().__init__()
#         self.ffn_type = ffn_type
#         hidden_dim = 4 * dim
#         if ffn_type == "swiglu":
#             self.w1 = nn.Linear(dim, hidden_dim, bias=False)
#             self.w2 = nn.Linear(hidden_dim, dim, bias=False)
#             self.w3 = nn.Linear(dim, hidden_dim, bias=False)
#         else:  # Standard GPT-2 GELU FFN for ablation comparison
#             self.c_fc = nn.Linear(dim, hidden_dim)
#             self.c_proj = nn.Linear(hidden_dim, dim)
#         self.dropout = nn.Dropout(DROPOUT)

#     def forward(self, x):
#         if self.ffn_type == "swiglu":
#             return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
#         return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


# class Block(nn.Module):
#     def __init__(self, dim, n_heads, max_seq_len, use_rope=True, norm_type="rmsnorm", ffn_type="swiglu"):
#         super().__init__()
#         norm_cls = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
#         self.ln1 = norm_cls(dim)
#         self.attn = CausalSelfAttention(dim, n_heads, max_seq_len, use_rope=use_rope)
#         self.ln2 = norm_cls(dim)
#         self.ffn = FeedForward(dim, ffn_type=ffn_type)

#     def forward(self, x, cos, sin, start_pos=0, kv_cache=None, use_cache=False):
#         attn_out, new_kv_cache = self.attn(
#             self.ln1(x), cos, sin, start_pos=start_pos, kv_cache=kv_cache, use_cache=use_cache
#         )
#         x = x + attn_out
#         x = x + self.ffn(self.ln2(x))
#         return x, new_kv_cache


# class ModernLLM(nn.Module):
#     def __init__(
#         self,
#         vocab_size,
#         dim,
#         n_heads,
#         n_layers,
#         context_length,
#         use_rope=True,
#         norm_type="rmsnorm",
#         ffn_type="swiglu",
#         tie_weights=TIE_WEIGHTS,
#     ):
#         super().__init__()
#         self.context_length = context_length
#         self.use_rope = use_rope

#         self.token_embedding = nn.Embedding(vocab_size, dim)
#         nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

#         self.pos_embedding = None if use_rope else nn.Embedding(context_length, dim)
#         if self.pos_embedding is not None:
#             nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

#         self.blocks = nn.ModuleList(
#             [
#                 Block(dim, n_heads, context_length, use_rope=use_rope, norm_type=norm_type, ffn_type=ffn_type)
#                 for _ in range(n_layers)
#             ]
#         )
#         norm_cls = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
#         self.ln_f = norm_cls(dim)
#         self.lm_head = nn.Linear(dim, vocab_size, bias=False)

#         if tie_weights:
#             self.lm_head.weight = self.token_embedding.weight

#         cos, sin = precompute_rope_freqs(dim // n_heads, context_length)
#         self.register_buffer("cos", cos, persistent=False)
#         self.register_buffer("sin", sin, persistent=False)

#     def forward(self, idx, targets=None, start_pos=0, kv_caches=None, use_cache=False):
#         B, T = idx.shape
#         x = self.token_embedding(idx)

#         if not self.use_rope and self.pos_embedding is not None:
#             positions = torch.arange(start_pos, start_pos + T, device=idx.device)
#             x = x + self.pos_embedding(positions)

#         new_kv_caches = [] if use_cache else None
#         for i, block in enumerate(self.blocks):
#             layer_cache = kv_caches[i] if kv_caches is not None else None
#             x, new_cache = block(
#                 x, self.cos, self.sin, start_pos=start_pos, kv_cache=layer_cache, use_cache=use_cache
#             )
#             if use_cache:
#                 new_kv_caches.append(new_cache)

#         x = self.ln_f(x)
#         logits = self.lm_head(x)

#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(
#                 logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
#             )

#         if use_cache:
#             return logits, loss, new_kv_caches
#         return logits, loss

#     @torch.no_grad()
#     def generate(
#         self,
#         idx,
#         max_new_tokens,
#         temperature=0.65,
#         top_k=30,
#         top_p=0.85,
#         repetition_penalty=1.12,
#         use_cache=True,
#         eot_token=50256,
#     ):
#         kv_caches = None
#         for step in range(max_new_tokens):
#             # Crop and reset cache if sequence exceeds context_length
#             if idx.size(1) > self.context_length:
#                 idx_cond = idx[:, -self.context_length :]
#                 kv_caches = None
#             else:
#                 idx_cond = idx

#             if use_cache and kv_caches is not None:
#                 start_pos = idx_cond.size(1) - 1
#                 logits, _, kv_caches = self(
#                     idx_cond[:, -1:], start_pos=start_pos, kv_caches=kv_caches, use_cache=True
#                 )
#             elif use_cache:
#                 logits, _, kv_caches = self(
#                     idx_cond, start_pos=0, kv_caches=None, use_cache=True
#                 )
#             else:
#                 logits, _ = self(idx_cond, start_pos=0, use_cache=False)

#             logits = logits[:, -1, :]

#             if repetition_penalty != 1.0:
#                 local_window = idx_cond[0, -24:]
#                 recent_tokens = torch.unique(local_window)
#                 recent_logits = logits[0, recent_tokens]
#                 logits[0, recent_tokens] = torch.where(
#                     recent_logits < 0,
#                     recent_logits * repetition_penalty,
#                     recent_logits / repetition_penalty,
#                 )

#             logits = logits / max(temperature, 1e-5)

#             if top_k is not None:
#                 v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#                 logits[logits < v[:, [-1]]] = float("-inf")

#             if top_p is not None and 0.0 < top_p < 1.0:
#                 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#                 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#                 sorted_indices_to_remove = cumulative_probs > top_p
#                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#                 sorted_indices_to_remove[..., 0] = 0
#                 indices_to_remove = sorted_indices_to_remove.scatter(
#                     1, sorted_indices, sorted_indices_to_remove
#                 )
#                 logits[indices_to_remove] = float("-inf")

#             probs = F.softmax(logits, dim=-1)
#             idx_next = torch.multinomial(probs, num_samples=1)
#             idx = torch.cat((idx, idx_next), dim=1)

#             if eot_token is not None and idx_next.item() == eot_token:
#                 break

#         return idx

