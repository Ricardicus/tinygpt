import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters: γ (scale) and β (bias)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        """
        x: (batch_size, context_length, d_model)
        """
        return F.layer_norm(x, (self.d_model,), self.gamma, self.beta, self.eps)


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Learnable projection matrices
        self.W_q = nn.Parameter(
            torch.randn(d_model, d_model) * (1 / math.sqrt(d_model))
        )
        self.W_k = nn.Parameter(
            torch.randn(d_model, d_model) * (1 / math.sqrt(d_model))
        )
        self.W_v = nn.Parameter(
            torch.randn(d_model, d_model) * (1 / math.sqrt(d_model))
        )

    def forward(self, x):
        """
        x: (B, C, d_model)
        """
        B, C, d = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)

        # causal mask: prevent looking ahead
        mask = torch.tril(torch.ones(C, C, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights @ V  # (B, C, d_model)

        return attn_out


class QKRMSNorm(nn.Module):
    """Normalize each head over its features, sharing learned scales across heads."""
    def __init__(self, d_head, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_head))
        self.eps = eps

    def forward(self, x):
        # Accumulate statistics in FP32, then preserve the attention kernel dtype.
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(x.dtype)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, qk_norm=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Combined projection for Q, K, V (for efficiency)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        # Output projection after concatenating all heads
        self.W_o = nn.Linear(d_model, d_model)
        self.q_norm = QKRMSNorm(self.d_head) if qk_norm else nn.Identity()
        self.k_norm = QKRMSNorm(self.d_head) if qk_norm else nn.Identity()

    def forward(self, x):
        B, C, D = x.shape  # (batch, context_len, d_model)

        # Project once, then split into Q, K, V
        qkv = self.W_qkv(x)  # (B, C, 3*d_model)
        qkv = qkv.reshape(B, C, 3, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, C, d_head)
        Q, K = self.q_norm(Q), self.k_norm(K)

        # PyTorch selects an optimized causal attention kernel for this device/dtype.
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Recombine heads
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, C, D)

        # Final linear projection
        return self.W_o(attn_out)


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        hidden = expansion * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_heads, qk_norm=False):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, d_heads, qk_norm=qk_norm)
        self.ln2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self, vocab_size=1920, context_length=256, d_model=512, n_heads=8, num_layers=6,
        qk_norm=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.qk_norm = qk_norm

        # Embeddings
        self.E = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.pos_E = nn.Parameter(torch.randn(context_length, d_model) * 0.02)

        # Stack of transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, qk_norm=qk_norm) for _ in range(num_layers)]
        )

        # Each block contributes two residual branches. Scale their output
        # projections at initialization to limit growth across deep stacks.
        if num_layers:
            residual_scale = 1 / math.sqrt(2 * num_layers)
            with torch.no_grad():
                for layer in self.layers:
                    for projection in (layer.attn.W_o, layer.mlp.fc2):
                        projection.weight.mul_(residual_scale)
                        if projection.bias is not None:
                            projection.bias.zero_()

        # Final normalization
        self.final_ln = LayerNorm(d_model)

    def forward(self, token_ids):
        B, C = token_ids.shape

        # Input embedding
        tok_emb = F.embedding(token_ids, self.E)
        pos_emb = self.pos_E[:C]
        x = tok_emb + pos_emb

        # Pass through N transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        x = self.final_ln(x)

        # Weight-tied projection to logits
        logits = x @ self.E.T
        return logits
