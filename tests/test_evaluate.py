import math
from types import SimpleNamespace

import pytest
import torch

from scglm.evaluate import (
    build_harness_command, evaluate_documents, fixed_mixture_metrics,
    score_document, wikitext103_documents,
)






















class PredictNext(torch.nn.Module):
    def __init__(self, vocab=7, context=4):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab, max_position_embeddings=context)

    def forward(self, input_ids, use_cache=False):
        logits = torch.zeros((*input_ids.shape, self.config.vocab_size), device=input_ids.device)
        logits.scatter_(-1, ((input_ids + 1) % self.config.vocab_size).unsqueeze(-1), 3.0)
        return SimpleNamespace(logits=logits)


def test_sliding_evaluation_scores_every_target_once_and_single_shift():
    model = PredictNext()
    tokens = [i % 7 for i in range(13)]
    loss, count = score_document(model, tokens, context_length=4, stride=2)
    expected = math.log(math.exp(3) + 6) - 3
    assert count == 12
    assert loss == pytest.approx(count * expected, rel=1e-6)
    truncated_loss, truncated_count = score_document(model, tokens, context_length=4, stride=2,
                                                     max_predictable_tokens=5)
    assert truncated_count == 5
    assert truncated_loss == pytest.approx(5 * expected, rel=1e-6)


def test_document_isolation_mode_restore_and_fixed_q():
    model = PredictNext()
    model.train()
    result = evaluate_documents(model, [{"id": "a", "source": "web", "tokens": [0, 1, 2]},
                                         {"id": "b", "source": "wiki", "input_ids": [4, 5]}],
                                context_length=4, stride=2)
    assert model.training is True
    assert [r["token_count"] for r in result] == [2, 1]
    metrics = fixed_mixture_metrics(result)
    assert metrics["fixed_q_nll"] == pytest.approx(math.log(math.exp(3) + 6) - 3, rel=1e-6)
    with pytest.raises(ValueError):
        evaluate_documents(model, [{"id": "bad", "source": "web", "tokens": [0, 7]}],
                           context_length=4, stride=2)
    assert model.training is True


def test_official_evaluations_are_sealed_without_explicit_gate():
    with pytest.raises(ValueError, match="sealed"):
        wikitext103_documents(revision="not-a-revision", tokenizer_path="absent")
    with pytest.raises(ValueError, match="sealed"):
        build_harness_command(model_path="absent", harness_repo="absent", harness_commit="absent",
                              output_path="absent")
