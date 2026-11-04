"""Unit tests for the temporal Transformer baseline."""
import numpy as np
import pytest
import torch

from adapt_g.models.temporal import TemporalConfig, TemporalTransformer


def _dummy_batch(batch_size: int = 4, max_len: int = 16):
    torch.manual_seed(0)
    seq = torch.randn(batch_size, max_len, 4)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    # Mark first 4 positions as pad for half of the batch.
    mask[::2, :4] = False
    cur = torch.randn(batch_size, 4)
    return seq, mask, cur


def test_forward_shapes():
    cfg = TemporalConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = TemporalTransformer(cfg)
    seq, mask, cur = _dummy_batch()
    out = model(seq, mask, cur)
    assert out.shape == (seq.size(0),)


def test_padding_does_not_leak_into_pooling():
    """Rows that differ only in pad positions produce identical logits."""
    cfg = TemporalConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = TemporalTransformer(cfg).eval()
    seq, mask, cur = _dummy_batch(batch_size=2, max_len=8)
    # Row 0 has pads at [0:4]; row 1 has none. To compare, we make row 1 a copy
    # of row 0 at real positions and set its pad positions to a huge value.
    seq[1] = seq[0]
    seq[1, :4] = 999.0  # huge values at positions that are padded in row 0
    cur[1] = cur[0]
    mask[1, :4] = False
    with torch.no_grad():
        out = model(seq, mask, cur)
    assert torch.allclose(out[0], out[1], atol=1e-5)


def test_all_padded_row_does_not_crash():
    cfg = TemporalConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = TemporalTransformer(cfg).eval()
    seq, mask, cur = _dummy_batch(batch_size=2, max_len=8)
    mask[0] = False  # fully padded
    with torch.no_grad():
        out = model(seq, mask, cur)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_gradients_reach_every_component():
    cfg = TemporalConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32)
    model = TemporalTransformer(cfg)
    seq, mask, cur = _dummy_batch()
    out = model(seq, mask, cur).sum()
    out.backward()
    names_with_grad = {n for n, p in model.named_parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0}
    # Every submodule should receive gradient.
    for prefix in ("step_proj", "cur_mlp", "head", "encoder"):
        assert any(n.startswith(prefix) for n in names_with_grad), prefix


def test_parameter_count_positive():
    cfg = TemporalConfig(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    m = TemporalTransformer(cfg)
    assert m.n_parameters() > 0
    # Sanity: dominated by encoder layers.
    assert m.n_parameters() > cfg.d_model * cfg.d_model


def test_deterministic_with_fixed_seed():
    cfg = TemporalConfig(d_model=16, n_heads=2, n_layers=1, d_ff=32)
    torch.manual_seed(42)
    m1 = TemporalTransformer(cfg).eval()
    torch.manual_seed(42)
    m2 = TemporalTransformer(cfg).eval()
    seq, mask, cur = _dummy_batch()
    with torch.no_grad():
        o1 = m1(seq, mask, cur)
        o2 = m2(seq, mask, cur)
    assert torch.allclose(o1, o2, atol=1e-6)