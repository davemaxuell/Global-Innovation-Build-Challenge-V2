"""Document-level likelihood evaluation with an explicit final-evaluation gate.

Development JSONL records contain {id, source, tokens: [int, ...]}; input_ids is
accepted as an alias. Each document's first token is unscored. Sliding windows
score every subsequent token once and never cross document boundaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


def load_documents(paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    documents = []
    for path in paths:
        with Path(path).open() as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    record = json.loads(line)
                    if not all(k in record for k in ("id", "source")):
                        raise ValueError(f"Missing document identity in {path}:{line_number}")
                    documents.append(record)
    return documents


def _device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.inference_mode()
def score_document(model: torch.nn.Module, tokens: Sequence[int], *, context_length: int = 1024,
                   stride: int = 512, max_predictable_tokens: int | None = None,
                   device: str | torch.device | None = None) -> tuple[float, int]:
    """Return summed NLL and scored tokens; loss is FP32, accumulation FP64."""
    if context_length < 2 or not 1 <= stride < context_length:
        raise ValueError("Require context_length >= 2 and 1 <= stride < context_length")
    model_context = getattr(getattr(model, "config", None), "max_position_embeddings", context_length)
    if context_length > model_context:
        raise ValueError("Evaluation context exceeds the model's configured context")
    if max_predictable_tokens is not None and max_predictable_tokens <= 0:
        raise ValueError("max_predictable_tokens must be positive")
    ids = list(tokens[:max_predictable_tokens + 1] if max_predictable_tokens is not None else tokens)
    if any(isinstance(t, bool) or not isinstance(t, Integral) or t < 0 for t in ids):
        raise ValueError("Token IDs must be nonnegative integers")
    ids = [int(t) for t in ids]
    vocabulary = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocabulary is not None and any(t >= vocabulary for t in ids):
        raise ValueError("Document contains an out-of-vocabulary token")
    if len(ids) < 2:
        return 0.0, 0
    target_device = torch.device(device) if device is not None else _device(model)
    total_nll, token_count, previous_end = 0.0, 0, 1
    for start in range(0, len(ids), stride):
        end = min(start + context_length, len(ids))
        if end <= previous_end:
            break
        inputs = torch.tensor([ids[start:end]], dtype=torch.long, device=target_device)
        outputs = model(input_ids=inputs, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        # logits[:, i] predicts inputs[:, i+1]. Never rely on model labels or
        # model-provided loss here, which could shift an already-shifted target.
        first_target = max(previous_end, start + 1)
        offset = first_target - start - 1
        selected_logits = logits[:, offset:-1, :].float()
        targets = inputs[:, offset + 1:]
        losses = F.cross_entropy(selected_logits.reshape(-1, selected_logits.shape[-1]),
                                 targets.reshape(-1), reduction="none")
        if not torch.isfinite(losses).all():
            raise ValueError("Nonfinite evaluation loss")
        total_nll += losses.double().sum().item()
        token_count += targets.numel()
        previous_end = end
        if end == len(ids):
            break
    if token_count != len(ids) - 1:
        raise AssertionError("Sliding-window scorer failed the exact-once token contract")
    return total_nll, token_count


def evaluate_documents(model: torch.nn.Module, documents: Iterable[Mapping[str, Any]], *,
                       context_length: int = 1024, stride: int = 512,
                       max_predictable_tokens: int | None = None,
                       device: str | torch.device | None = None) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    records, seen = [], set()
    try:
        for document in documents:
            key = (str(document["source"]), str(document["id"]))
            if key in seen:
                raise ValueError(f"Duplicate document identity: {key}")
            seen.add(key)
            tokens = document.get("tokens", document.get("input_ids"))
            if tokens is None:
                raise ValueError("A document requires tokens or input_ids")
            loss, count = score_document(model, tokens, context_length=context_length, stride=stride,
                                          max_predictable_tokens=max_predictable_tokens, device=device)
            records.append({"id": key[1], "source": key[0], "nll_sum": loss, "token_count": count})
    finally:
        model.train(was_training)
    return records


eval_documents = evaluate_documents


def fixed_mixture_metrics(records: Sequence[Mapping[str, Any]],
                          weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    weights = dict(weights or {"web": 0.8, "wiki": 0.2})
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Fixed evaluation weights must be positive and sum to one")
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"nll_sum": 0.0, "token_count": 0, "documents": 0})
    for record in records:
        loss, count = float(record["nll_sum"]), int(record["token_count"])
        if not math.isfinite(loss) or loss < 0 or count < 0 or count != record["token_count"]:
            raise ValueError("Invalid document evaluation record")
        source = str(record["source"])
        if source not in weights:
            raise ValueError(f"Unexpected evaluation source: {source}")
        grouped[source]["nll_sum"] += loss
        grouped[source]["token_count"] += count
        grouped[source]["documents"] += 1
    for source in weights:
        if grouped[source]["token_count"] <= 0:
            raise ValueError(f"No scored tokens for {source}")
        nll = grouped[source]["nll_sum"] / grouped[source]["token_count"]
        grouped[source].update(nll=nll, perplexity=math.exp(nll) if nll < 709 else None)
    loss = sum(weights[s] * grouped[s]["nll"] for s in weights)
    return {"fixed_q_nll": loss, "fixed_q_perplexity": math.exp(loss) if loss < 709 else None,
            "weights": weights, "sources": dict(grouped)}


def write_evaluation_report(path: str | Path, records: Sequence[Mapping[str, Any]], *,
                            metadata: Mapping[str, Any] | None = None,
                            weights: Mapping[str, float] | None = None) -> dict[str, Any]:
    report = {"schema_version": 1, "metadata": dict(metadata or {}),
              "metrics": fixed_mixture_metrics(records, weights), "documents": list(records)}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return report


def _encode_text(tokenizer_path: str | Path):
    from tokenizers import Tokenizer
    path = Path(tokenizer_path)
    if path.is_dir():
        path = path / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def wikitext103_documents(*, revision: str, tokenizer_path: str | Path,
                          split: str = "validation", final_evaluation: bool = False) -> list[dict[str, Any]]:
    """Load the explicit WikiText-103 RAW corpus, grouped by article headings."""
    if not final_evaluation:
        raise ValueError("Official evaluation is sealed until model selection; require --final-evaluation")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Pin the dataset revision to a full 40-character commit hash")
    if split not in {"validation", "test"}:
        raise ValueError("Official perplexity requires a held-out split")
    from datasets import load_dataset
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, revision=revision)
    encode = _encode_text(tokenizer_path)
    documents, paragraphs = [], []
    heading = re.compile(r"^\s*=\s+[^=].*?\s+=\s*$")

    def append_article() -> None:
        if not paragraphs:
            return
        text = "\n".join(paragraphs)
        identity = hashlib.sha256(text.encode("utf-8")).hexdigest()
        documents.append({"id": identity, "source": "wikitext103", "tokens": encode(text)})

    for row in dataset:
        paragraph = row["text"]
        if heading.match(paragraph.strip()) and paragraphs:
            append_article()
            paragraphs = []
        if paragraph.strip():
            paragraphs.append(paragraph)
    append_article()
    if not documents:
        raise ValueError("WikiText-103 produced no article documents")
    return documents


def build_harness_command(*, model_path: str | Path, harness_repo: str | Path,
                          harness_commit: str, output_path: str | Path, device: str = "cpu",
                          final_evaluation: bool = False) -> dict[str, Any]:
    """Prepare, never execute, a command against a checked-out pinned harness."""
    if not final_evaluation:
        raise ValueError("Official benchmark scores remain sealed until model selection")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", harness_commit):
        raise ValueError("Pin a full harness commit hash")
    repo, model = Path(harness_repo).resolve(), Path(model_path).resolve()
    if not model.is_dir():
        raise ValueError("Use a local exported model directory")
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if actual.lower() != harness_commit.lower() or dirty:
        raise ValueError("Harness checkout must be clean and match the pinned commit")
    return {"cwd": str(repo), "harness_commit": actual,
            "command": [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args",
                        f"pretrained={model},trust_remote_code=False", "--tasks",
                        "hellaswag,arc_easy,piqa,winogrande", "--num_fewshot", "0", "--device", device,
                        "--batch_size", "1", "--output_path", str(Path(output_path).resolve()), "--log_samples"],
            "note": "Default pinned task splits/normalization must match the frozen protocol. "
                    "This command excludes WikiText: score WikiText-103 with the separate evaluator."}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("dev", help="Evaluate prepared development documents")
    wiki = commands.add_parser("wikitext103", help="Explicit final WikiText-103 token perplexity")
    for command in (dev, wiki):
        command.add_argument("--model", required=True, help="Our local HF export directory")
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--context-length", type=int, default=1024)
        command.add_argument("--stride", type=int, default=512)
    dev.add_argument("--documents", nargs="+", required=True)
    dev.add_argument("--max-predictable-tokens", type=int)
    wiki.add_argument("--tokenizer", required=True)
    wiki.add_argument("--dataset-revision", required=True)
    wiki.add_argument("--split", default="validation", choices=("validation", "test"))
    wiki.add_argument("--final-evaluation", action="store_true", help="Attest that model selection is already frozen")
    harness = commands.add_parser("harness-command", help="Print a pinned command; does not run benchmarks")
    harness.add_argument("--model", required=True)
    harness.add_argument("--harness-repo", required=True)
    harness.add_argument("--harness-commit", required=True)
    harness.add_argument("--output", required=True)
    harness.add_argument("--device", default="cpu")
    harness.add_argument("--final-evaluation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "harness-command":
        result = build_harness_command(model_path=args.model, harness_repo=args.harness_repo,
                                       harness_commit=args.harness_commit, output_path=args.output,
                                       device=args.device, final_evaluation=args.final_evaluation)
        print(json.dumps(result, indent=2))
        return
    if args.command == "wikitext103" and not args.final_evaluation:
        parser.error("Official evaluation is sealed: freeze model selection, then use --final-evaluation")
    from .model import load_model
    model = load_model(args.model).to(args.device)
    if args.command == "dev":
        documents = load_documents(args.documents)
        max_tokens, weights = args.max_predictable_tokens, None
        metadata = {"kind": "development", "document_paths": args.documents}
    else:
        documents = wikitext103_documents(revision=args.dataset_revision, tokenizer_path=args.tokenizer,
                                          split=args.split, final_evaluation=args.final_evaluation)
        max_tokens, weights = None, {"wikitext103": 1.0}
        metadata = {"kind": "final", "dataset": "Salesforce/wikitext", "configuration": "wikitext-103-raw-v1",
                    "revision": args.dataset_revision, "split": args.split, "article_boundary": "level-one heading",
                    "special_tokens": "none added", "first_document_token": "unscored"}
    records = evaluate_documents(model, documents, context_length=args.context_length, stride=args.stride,
                                  max_predictable_tokens=max_tokens, device=args.device)
    metadata.update(model=str(Path(args.model).resolve()), context_length=args.context_length,
                    stride=args.stride, max_predictable_tokens=max_tokens, scoring="token NLL; exact-once targets")
    report = write_evaluation_report(args.output, records, metadata=metadata, weights=weights)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
