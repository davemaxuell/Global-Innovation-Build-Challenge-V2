"""CPU-only checks for objective, causality, and export invariants."""

import pytest
import torch
from torch.nn import functional as F

from scglm.model import (
    EXPECTED_DEFAULT_PARAMETERS,
    audit_parameters,
    build_model,
    count_parameters,
    forward_loss,
    load_model,
    save_model,
    token_logits,
)


TINY_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 80,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 16,
    "seed": 123,
}


@pytest.fixture
def model():
    return build_model(TINY_CONFIG)


def test_future_tokens_cannot_change_prefix_logits(model):
    model.eval()
    original = torch.tensor([[1, 4, 5, 6, 7, 8]])
    changed = torch.tensor([[1, 4, 5, 21, 22, 23]])
    with torch.no_grad():
        original_logits = token_logits(model, original)
        changed_logits = token_logits(model, changed)
    torch.testing.assert_close(original_logits[:, :3], changed_logits[:, :3], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original_logits[:, 3:], changed_logits[:, 3:])


def test_external_shift_scores_every_supplied_target_without_second_shift(model):
    block = torch.tensor([[1, 5, 8, 13, 21], [2, 3, 7, 17, 31]])
    x, y = block[:, :-1], block[:, 1:]
    logits = token_logits(model, x)
    manual = -F.log_softmax(logits.float(), dim=-1).gather(-1, y[..., None]).mean()
    actual = forward_loss(model, x, y)
    torch.testing.assert_close(actual, manual)
    actual.backward()
    assert model.get_input_embeddings().weight.grad is not None
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
    # In particular, the final supplied target contributes to the loss.
    last_changed = y.clone()
    last_changed[:, -1] = torch.tensor([62, 63])
    assert not torch.isclose(actual.detach(), forward_loss(model, x, last_changed))


def test_tying_parameter_count_and_save_reload_predictions(model, tmp_path):
    # V*d + L*(4*d*d + 3*d*f + 2*d) + d, with all norm gains included.
    expected = 64 * 32 + 2 * (4 * 32 * 32 + 3 * 32 * 80 + 2 * 32) + 32
    assert count_parameters(model) == expected
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert audit_parameters(model)["analytic_parameters"] == expected
    # Exercise serialization of genuinely updated shared weights, not just init.
    x, y = torch.tensor([[1, 5, 8]]), torch.tensor([[5, 8, 13]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    forward_loss(model, x, y).backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected_logits = token_logits(model, x)
    save_model(model, tmp_path)
    restored = load_model(tmp_path).eval()
    assert count_parameters(restored) == expected
    assert restored.get_input_embeddings().weight is restored.get_output_embeddings().weight
    with torch.no_grad():
        torch.testing.assert_close(token_logits(restored, x), expected_logits, atol=1e-6, rtol=1e-6)


def test_architecture_and_context_contracts_fail_before_use(model):
    with pytest.raises(ValueError, match="share one parameter"):
        build_model({**TINY_CONFIG, "tie_word_embeddings": False})
    with pytest.raises(ValueError, match="exceeds cap"):
        build_model({**TINY_CONFIG, "max_parameters": 100})
    with pytest.raises(ValueError, match="exceeds the configured"):
        token_logits(model, torch.ones((1, 17), dtype=torch.long))
    with pytest.raises(ValueError, match="same externally shifted"):
        forward_loss(model, torch.ones((1, 4), dtype=torch.long), torch.ones((1, 3), dtype=torch.long))


def test_seeded_build_preserves_caller_rng_and_reproduces_weights():
    before = torch.random.get_rng_state().clone()
    first = build_model(TINY_CONFIG)
    assert torch.equal(torch.random.get_rng_state(), before)
    second = build_model(TINY_CONFIG)
    torch.testing.assert_close(first.get_input_embeddings().weight, second.get_input_embeddings().weight)


def test_full_default_architecture_cap_without_allocating_training_weights():
    # Meta construction checks the actual module shapes and sharing without a
    # GPU or a 46M-weight allocation. Tiny CPU tests above execute the math.
    with torch.device("meta"):
        default_model = build_model()
    assert count_parameters(default_model) == EXPECTED_DEFAULT_PARAMETERS
    assert audit_parameters(default_model)["headroom"] == 3_653_248
