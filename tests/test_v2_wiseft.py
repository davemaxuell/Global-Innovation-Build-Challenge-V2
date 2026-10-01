"""WiSE-FT interpolation: endpoints, linearity and tied weights."""
import torch

from scglm.model import build_model
from scglm_v2.wiseft import interpolate

TINY = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 80, "num_hidden_layers": 2,
        "num_attention_heads": 4, "max_position_embeddings": 32, "seed": 123}


def _pair():
    a = build_model({**TINY, "seed": 1})
    b = build_model({**TINY, "seed": 2})
    return a, b


def test_endpoints_and_midpoint():
    a, b = _pair()
    pa = {k: v.clone() for k, v in a.named_parameters()}
    pb = {k: v.clone() for k, v in b.named_parameters()}
    zero = interpolate(build_model({**TINY, "seed": 1}), b, 0.0)
    assert all(torch.equal(p, pa[k]) for k, p in zero.named_parameters())
    one = interpolate(build_model({**TINY, "seed": 1}), b, 1.0)
    assert all(torch.allclose(p, pb[k]) for k, p in one.named_parameters())
    half = interpolate(a, b, 0.5)
    assert all(torch.allclose(p, (pa[k] + pb[k]) / 2, atol=1e-7) for k, p in half.named_parameters())
    assert half.lm_head.weight.data_ptr() == half.model.embed_tokens.weight.data_ptr()   # still tied
