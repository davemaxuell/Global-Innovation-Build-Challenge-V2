"""Text-completion demo of the V2 scratch-pretrained model; loads local files only.

V2 is a base language model (no instruction tuning), so the prompt is continued as
plain text. Greedy decoding by default; ``--sample`` switches to temperature/top-p.

Run: PYTHONPATH=src python scripts/demo.py --prompt "The water cycle begins when"
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from transformers import AutoTokenizer
    from scglm.model import load_model

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints/v2_best")
    p.add_argument("--prompt", default="The purpose of a language model is")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not 1 <= a.max_new_tokens <= 512:
        p.error("Choose 1-512 generated tokens")
    model = load_model(a.model).to(a.device).eval()  # refuses anything but a local scratch-trained export
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    provenance = json.loads((a.model / "training_provenance.json").read_text())
    ids = tokenizer.encode(a.prompt, add_special_tokens=False)
    if not ids or len(ids) + a.max_new_tokens > model.config.max_position_embeddings:
        p.error("Prompt and requested continuation must fit the 1,024-token context")
    torch.manual_seed(a.seed)
    inputs = torch.tensor([[tokenizer.eos_token_id] + ids], device=a.device)  # EOS prefix, as in training/evaluation
    kwargs = dict(do_sample=a.sample, max_new_tokens=a.max_new_tokens, repetition_penalty=a.repetition_penalty,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if a.sample:
        kwargs.update(temperature=a.temperature, top_p=a.top_p)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
    print(f"[model: {a.model.name} · {provenance.get('cumulative_tokens', provenance.get('main_tokens')) / 1e9:.1f}B "
          f"pretraining tokens · {sum(x.numel() for x in model.parameters()):,} parameters]")
    print(a.prompt + tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
