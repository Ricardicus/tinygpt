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
        # Compute mean and variance along the last dimension (features)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * x_hat + self.beta
        return out

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Learnable projection matrices
        self.W_q = nn.Parameter(torch.randn(d_model, d_model) * (1 / math.sqrt(d_model)))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model) * (1 / math.sqrt(d_model)))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model) * (1 / math.sqrt(d_model)))

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
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_out = attn_weights @ V  # (B, C, d_model)

        return attn_out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Combined projection for Q, K, V (for efficiency)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        # Output projection after concatenating all heads
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, C, D = x.shape  # (batch, context_len, d_model)

        # Project once, then split into Q, K, V
        qkv = self.W_qkv(x)  # (B, C, 3*d_model)
        qkv = qkv.reshape(B, C, 3, self.num_heads, self.d_head).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, C, d_head)

        # Compute attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, num_heads, C, C)

        # Apply causal mask (prevent attending to future tokens)
        mask = torch.tril(torch.ones(C, C, device=x.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and attention output
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = attn_weights @ V  # (B, num_heads, C, d_head)

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
    def __init__(self, d_model, d_heads):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, d_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size=1920, context_length=256, d_model=512, n_heads=8, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_heads = n_heads

        # Embeddings
        self.E = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.pos_E = nn.Parameter(torch.randn(context_length, d_model) * 0.02)

        # Stack of transformer layers
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(num_layers)])

    def forward(self, token_ids):
        B, C = token_ids.shape

        # Input embedding
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float()
        tok_emb = one_hot @ self.E
        pos_emb = self.pos_E[:C]
        x = tok_emb + pos_emb

        # Pass through N transformer layers
        for layer in self.layers:
            x = layer(x)

        return x
