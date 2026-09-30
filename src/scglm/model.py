"""A small Llama-family model initialized from configuration, never remote weights.

``forward_loss`` deliberately does not use the Transformers ``labels`` argument:
our data loader creates x=tokens[:-1], y=tokens[1:] before calling it. Shifting
inside the model as well would train the wrong objective.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaForCausalLM


PARAMETER_CAP = 50_000_000
EXPECTED_DEFAULT_PARAMETERS = 46_346_752
DEFAULT_CONFIG: dict[str, Any] = {
    "vocab_size": 16_384,
    "hidden_size": 512,
    "intermediate_size": 1_376,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "max_position_embeddings": 1_024,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
    "tie_word_embeddings": True,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "use_cache": False,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def _validate_config(config: LlamaConfig, cap: int) -> int:
    """Check the supported dense-MHA contract before allocating its weights."""
    for field in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "max_position_embeddings",
    ):
        value = getattr(config, field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not 0 < cap <= PARAMETER_CAP:
        raise ValueError(f"max_parameters must be within 1..{PARAMETER_CAP}")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("hidden_size must divide evenly into attention heads")
    if config.num_key_value_heads != config.num_attention_heads:
        raise ValueError("This experiment requires full multi-head attention")
    head_dim = config.hidden_size // config.num_attention_heads
    if config.head_dim != head_dim or head_dim % 2:
        raise ValueError("RoPE requires an even head_dim equal to hidden_size / heads")
    if config.hidden_act != "silu":
        raise ValueError("The fixed architecture requires SwiGLU (hidden_act='silu')")
    if not config.tie_word_embeddings:
        raise ValueError("Input and output embeddings must share one parameter")
    if config.attention_bias or config.mlp_bias:
        raise ValueError("The fixed architecture has no attention or MLP biases")
    if config.attention_dropout != 0.0:
        raise ValueError("The fixed experiment uses zero attention dropout")
    rope = config.rope_parameters
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Only unscaled default RoPE is supported")
    if rope.get("rope_theta") != 10_000.0:
        raise ValueError("The fixed experiment uses RoPE base 10000")
    for field in ("pad_token_id", "bos_token_id", "eos_token_id"):
        ids = getattr(config, field)
        ids = ids if isinstance(ids, list) else [ids]
        if any(value is not None and not 0 <= value < config.vocab_size for value in ids):
            raise ValueError(f"{field} must be inside the configured vocabulary")
    d, f = config.hidden_size, config.intermediate_size
    expected = config.vocab_size * d + config.num_hidden_layers * (
        4 * d * d + 3 * d * f + 2 * d
    ) + d
    if expected > cap:
        raise ValueError(f"Planned parameter count {expected:,} exceeds cap {cap:,}")
    return expected


def count_parameters(model: nn.Module) -> int:
    """Count unique trainable Parameter objects, counting tied weights once."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def audit_parameters(model: LlamaForCausalLM, max_parameters: int = PARAMETER_CAP) -> dict[str, Any]:
    """Check construction/export invariants and return a JSON-serializable audit."""
    expected = _validate_config(model.config, max_parameters)
    shared = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    if not shared:
        raise ValueError("Output and input embeddings are not the same Parameter")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = count_parameters(model)
    if total != expected or trainable != expected:
        raise ValueError(
            f"Parameter audit failed: analytic={expected}, total={total}, trainable={trainable}"
        )
    return {
        "trainable_parameters": trainable,
        "total_unique_parameters": total,
        "analytic_parameters": expected,
        "parameter_cap": max_parameters,
        "headroom": max_parameters - trainable,
        "tied_embeddings": shared,
    }


def build_model(config: dict[str, Any] | None = None) -> LlamaForCausalLM:
    """Build random FP32 weights from a flat configuration; download nothing.

    Dimension and special-token overrides support small correctness tests and a
    frozen tokenizer. ``seed`` seeds construction without changing caller RNG
    state. ``attn_implementation`` selects ``sdpa`` (default) or ``eager``.
    ``max_parameters`` can tighten, but cannot enlarge, the competition cap.
    """
    overrides = dict(config or {})
    seed = overrides.pop("seed", None)
    backend = overrides.pop("attn_implementation", "sdpa")
    cap = overrides.pop("max_parameters", PARAMETER_CAP)
    # Accept the common earlier Transformers spelling in our public config.
    if "rope_theta" in overrides:
        if "rope_parameters" in overrides:
            raise ValueError("Specify rope_theta or rope_parameters, not both")
        overrides["rope_parameters"] = {
            "rope_type": "default", "rope_theta": overrides.pop("rope_theta")
        }
    supported = set(DEFAULT_CONFIG) | {"head_dim"}
    unknown = set(overrides) - supported
    if unknown:
        raise ValueError(f"Unknown model configuration keys: {sorted(unknown)}")
    if backend not in {"sdpa", "eager"}:
        raise ValueError("attn_implementation must be 'sdpa' or 'eager'")
    values = {**DEFAULT_CONFIG, **overrides}
    if "num_attention_heads" in overrides and "num_key_value_heads" not in overrides:
        values["num_key_value_heads"] = values["num_attention_heads"]
    hf_config = LlamaConfig(**values)
    _validate_config(hf_config, cap)
    hf_config._attn_implementation = backend
    hf_config.scglm_initialization = "random"
    hf_config.scglm_initialization_seed = seed
    rng_context = torch.random.fork_rng(devices=[]) if seed is not None else nullcontext()
    with rng_context:
        if seed is not None:
            # Only CPU RNG: construction stays on CPU until the caller places it.
            torch.random.default_generator.manual_seed(seed)
        model = LlamaForCausalLM(hf_config).float()
    audit_parameters(model, cap)
    return model


def token_logits(model: LlamaForCausalLM, input_ids: Tensor) -> Tensor:
    """Return causal logits without shifting tokens or allocating a KV cache."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [batch, nonempty sequence]")
    if input_ids.shape[1] > model.config.max_position_embeddings:
        raise ValueError("Input exceeds the configured training/evaluation context")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_ids must contain integer token IDs")
    return model(input_ids=input_ids, use_cache=False, return_dict=True).logits


def forward_loss(model: LlamaForCausalLM, x: Tensor, y: Tensor) -> Tensor:
    """Mean next-token CE for already-shifted x/y; y may mask tokens with -100.

    For a block of T+1 tokens, pass ``x=block[:-1]`` and ``y=block[1:]``.
    All T logits predict their corresponding supplied targets exactly once.
    Caller must supply at least one unmasked target. Raw likelihood evaluation
    should use ``token_logits`` to keep token counts and reductions explicit.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same externally shifted [batch, sequence] shape")
    if y.dtype != torch.int64:
        raise TypeError("Cross-entropy targets must use torch.int64")
    logits = token_logits(model, x)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        ignore_index=-100,
    )


def save_model(model: LlamaForCausalLM, directory: str | Path) -> dict[str, Any]:
    """Export standard HF config/safetensors; save the tokenizer alongside it."""
    audit = audit_parameters(model)
    model.save_pretrained(str(directory), safe_serialization=True)
    return audit


def load_model(directory: str | Path, *, attn_implementation: str = "sdpa") -> LlamaForCausalLM:
    """Reload one of our local scratch-run exports; never resolve remote IDs."""
    path = Path(directory)
    if not path.is_dir():
        raise ValueError("load_model requires an existing local checkpoint directory")
    config = LlamaConfig.from_pretrained(str(path), local_files_only=True)
    if getattr(config, "scglm_initialization", None) != "random":
        raise ValueError("Checkpoint lacks this project's scratch-initialization marker")
    _validate_config(config, PARAMETER_CAP)
    if attn_implementation not in {"sdpa", "eager"}:
        raise ValueError("attn_implementation must be 'sdpa' or 'eager'")
    model = LlamaForCausalLM.from_pretrained(
        str(path), config=config, local_files_only=True,
        attn_implementation=attn_implementation, dtype=torch.float32,
    )
    audit_parameters(model)
    return model
