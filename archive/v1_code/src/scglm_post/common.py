"""Shared, explicit data and persistence contracts for local post-training."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from validation_history import ValidationRun, append_event, sha256, write_json

FORMAT_VERSION = "scglm-system-user-assistant-v2"
# Only existing tokens are used; encoding verifies the prefix boundary exactly.
CHAT_TEMPLATE = (
    "{% set offset = 1 if messages and messages[0]['role'] == 'system' else 0 %}"
    "{% if messages|length <= offset %}{{ raise_exception('Require a user turn') }}{% endif %}"
    "{% for message in messages %}"
    "{% if loop.index0 == 0 and offset == 1 %}{{ 'System:\\n' }}"
    "{% else %}"
    "{% set expected = 'user' if (loop.index0 - offset) % 2 == 0 else 'assistant' %}"
    "{% if message['role'] != expected %}{{ raise_exception('Invalid role order') }}{% endif %}"
    "{{ 'User:\\n' if expected == 'user' else 'Assistant:\\n' }}{% endif %}"
    "{% if message['content'] is not string or not message['content']|trim %}"
    "{{ raise_exception('Empty message') }}{% endif %}"
    "{{ message['content'] }}{% if message['role'] == 'assistant' %}{{ eos_token }}"
    "{% else %}{{ '\\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}"
    "{% if messages[-1]['role'] != 'user' %}{{ raise_exception('Generation requires a user turn') }}{% endif %}"
    "{{ 'Assistant:\\n' }}{% endif %}"
)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def identity(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def pretraining_tokens(provenance):
    for key in ("cumulative_pretraining_tokens", "cumulative_tokens", "main_tokens"):
        if key in provenance:
            value = provenance[key]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid pretraining token count")
            return value
    raise ValueError("Parent has no pretraining token count")


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def publish_export(temporary, destination):
    """Recover a crash between publishing an identical export and writing status."""
    import shutil
    temporary, destination = Path(temporary), Path(destination)
    if destination.exists():
        expected = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
        actual = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
        if actual != expected:
            raise ValueError("Existing export differs from the recovered final checkpoint")
        shutil.rmtree(temporary)
    else:
        temporary.rename(destination)


def tokenizer_identity(model_path, manifest):
    """Accept only the known HF no-op post-processor serialization difference.

    PreTrainedTokenizerFast.save_pretrained may replace a null post_processor
    with explicit identity TemplateProcessing. All vocabulary, merges, token
    IDs, normalization, decoding and other settings must remain exactly equal.
    Parent files are never rewritten. This does not permit arbitrary changes.
    """
    path = Path(model_path) / "tokenizer.json"
    actual = sha256(path)
    expected = manifest["tokenizer_sha256"]
    if actual == expected:
        return {"serialized_sha256": actual, "training_sha256": expected, "identity_template_only": False}
    source = manifest.get("config", {}).get("tokenizer")
    if not source:
        raise ValueError("Tokenizer mismatch without the frozen source")
    source = ROOT / source / "tokenizer.json"
    if sha256(source) != expected:
        raise ValueError("Frozen tokenizer source changed")
    left, right = json.loads(source.read_text()), json.loads(path.read_text())
    identity_template = {"type": "TemplateProcessing", "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                         "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                         "special_tokens": {}}
    if left.get("post_processor") is not None or right.get("post_processor") != identity_template:
        raise ValueError("Tokenizer changed beyond the known identity post-processor")
    right["post_processor"] = None
    if left != right:
        raise ValueError("Tokenizer vocabulary or other tokenization behavior changed")
    return {"serialized_sha256": actual, "training_sha256": expected, "identity_template_only": True,
            "scope": "identical single-sequence token IDs with add_special_tokens=False, as used by training and evaluation"}


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_messages(messages):
    offset = int(bool(messages) and messages[0].get("role") == "system")
    if len(messages) - offset < 2 or (len(messages) - offset) % 2:
        raise ValueError("Require complete alternating user/assistant turns")
    for i, message in enumerate(messages):
        expected = "system" if i == 0 and offset else ("user" if (i - offset) % 2 == 0 else "assistant")
        if message["role"] != expected:
            raise ValueError("Invalid role order")
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty message")
        if re.search(r"(^|\n)\s*(System|User|Assistant):", content) or re.search(r"<\|(?:pad|bos|eos|unk)\|>", content):
            raise ValueError("Reserved delimiter in content")


def render_prompt(messages):
    """Render optional system context and a conversation ending in a user turn."""
    validate_messages([*messages, {"role": "assistant", "content": "placeholder"}])
    parts = []
    for message in messages:
        if message["role"] in ("system", "user"):
            parts.append(message["role"].title() + ":\n" + message["content"] + "\n")
        else:
            parts.append("Assistant:\n" + message["content"] + "<|eos|>")
    return "".join(parts) + "Assistant:\n"


def encode_example(messages, tokenizer, *, max_length=1024, max_response_tokens=256):
    """Mask every prompt/prefix, supervise only the last answer and EOS.

    Earlier assistant turns are context only, avoiding repeated supervision of
    shared ancestors when conversation paths are expanded. No answer truncation.
    Tokenize the complete rendered text once and verify the generation prefix.
    """
    validate_messages(messages)
    prompt = render_prompt(messages[:-1])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + messages[-1]["content"], add_special_tokens=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer merges across prompt/response boundary")
    ids = full_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    response_tokens = len(ids) - len(prompt_ids)
    if response_tokens < 2 or response_tokens > max_response_tokens or len(ids) > max_length:
        raise ValueError("Example outside complete-example length limits")
    # get_vocab_size can scan the vocabulary in some tokenizers releases.
    # Resolve it once per example, never once per token.
    vocab_size = len(tokenizer)
    if any(token < 0 or token >= vocab_size for token in ids):
        raise ValueError("Token outside vocabulary")
    return {"input_ids": ids, "labels": labels, "prompt_length": len(prompt_ids),
            "target_tokens": response_tokens, "processed_tokens": len(ids)}


def check_encoded(row, *, vocab_size, eos_token_id, max_length):
    ids, labels, boundary = row["input_ids"], row["labels"], row["prompt_length"]
    if len(ids) != len(labels) or not 0 < boundary < len(ids) <= max_length:
        raise ValueError("Bad encoded example shape")
    if any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError("Bad token ID")
    if labels[:boundary] != [-100] * boundary or labels[boundary:] != ids[boundary:]:
        raise ValueError("Bad response-only mask")
    if ids[-1] != eos_token_id or row["target_tokens"] != len(ids) - boundary:
        raise ValueError("Bad EOS or target count")
    if row["processed_tokens"] != len(ids):
        raise ValueError("Bad processed-token count")


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest["status"] != "complete" or manifest["format_version"] not in (FORMAT_VERSION, "scglm-user-assistant-v1"):
        raise ValueError("Data preparation is incomplete or incompatible")
    for split, entry in manifest["splits"].items():
        if sha256(path.parent / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Data changed: {split}")
    return manifest
