import csv
import math

import torch

from main import write_layer_norms
from tinygpt import TinyGPT


def test_layer_monitoring_records_selected_tensors_and_appends(tmp_path):
    model = TinyGPT(vocab_size=16, context_length=4, d_model=8, n_heads=2, num_layers=4)
    model(torch.tensor([[1, 2, 3, 4]])).sum().backward()
    model.pos_E.grad = None
    path = tmp_path / "layers.csv"

    write_layer_norms(str(path), model, 1, 100)
    write_layer_norms(str(path), model, 1, 200)

    with path.open(newline="") as stream:
        blocks = list(csv.DictReader(stream))
    with (tmp_path / "layers.csv.parameters.csv").open(newline="") as stream:
        details = list(csv.DictReader(stream))

    assert len(blocks) == 8
    first = {row["parameter"]: row for row in details if row["global_step"] == "100"}
    assert {"E", "pos_E", "final_ln.gamma", "final_ln.beta"} <= first.keys()
    assert "layers.2.attn.W_qkv.weight" in first
    assert not any(name.startswith("layers.3.") for name in first)
    assert first["pos_E"]["gradient_norm"] == ""
    expected = model.layers[2].attn.W_qkv.weight.grad.norm().item()
    assert math.isclose(float(first["layers.2.attn.W_qkv.weight"]["gradient_norm"]), expected, rel_tol=1e-6)
    assert sum(row["parameter"] == "E" for row in details) == 2
