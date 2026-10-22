import unittest

import torch
import torch.nn.functional as F

from tinygpt import LayerNorm, MultiHeadSelfAttention, TinyGPT


class TransformerMathTests(unittest.TestCase):
    def test_layer_norm_matches_pytorch(self):
        norm = LayerNorm(12).double()
        x = torch.randn(2, 5, 12, dtype=torch.double)
        actual = norm(x)
        expected = ((x - x.mean(dim=-1, keepdim=True)) /
                    torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + norm.eps))
        expected = expected * norm.gamma + norm.beta
        torch.testing.assert_close(actual, expected)

    def test_attention_matches_pytorch_scaled_dot_product(self):
        attn = MultiHeadSelfAttention(12, 3).double()
        x = torch.randn(2, 5, 12, dtype=torch.double)
        q, k, v = attn.W_qkv(x).reshape(2, 5, 3, 3, 4).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / 2
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        expected = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1) @ v
        expected = attn.W_o(expected.transpose(1, 2).reshape(2, 5, 12))
        actual = attn(x)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.square().sum(), tuple(attn.parameters()), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), tuple(attn.parameters()))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_model_causality_and_finite_training_gradients(self):
        model = TinyGPT(vocab_size=20, context_length=8, d_model=12, n_heads=3, num_layers=2)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        changed = tokens.clone()
        changed[:, 3:] = torch.tensor([7, 8, 9])
        logits = model(tokens)
        torch.testing.assert_close(logits[:, :3], model(changed)[:, :3])
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 20), tokens[:, 1:].reshape(-1))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_embedding_lookup_matches_one_hot_and_gradients(self):
        tokens = torch.tensor([[1, 2, 1, 3]])
        old_weights = torch.randn(20, 12, requires_grad=True)
        new_weights = old_weights.detach().clone().requires_grad_(True)
        old = F.one_hot(tokens, num_classes=20).float() @ old_weights
        new = F.embedding(tokens, new_weights)
        torch.testing.assert_close(old, new)
        old.square().sum().backward()
        new.square().sum().backward()
        torch.testing.assert_close(old_weights.grad, new_weights.grad)


if __name__ == '__main__':
    unittest.main()
