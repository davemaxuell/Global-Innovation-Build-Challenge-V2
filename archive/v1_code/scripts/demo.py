"""Local demo of a selected scratch-trained model; never loads a remote model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))


def main():
    import torch
    from transformers import AutoTokenizer
    from scglm.model import load_model
    from scglm_post.common import render_prompt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prompt", default="The purpose of a language model is")
    p.add_argument("--system", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=128)
    a = p.parse_args()
    if not 1 <= a.max_new_tokens <= 512:
        p.error("Choose 1–512 generated tokens")
    model = load_model(a.model).to(a.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    provenance = json.loads((a.model/"training_provenance.json").read_text())
    stage = provenance.get("stage", "pretraining")
    if stage == "sft":
        messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
        prompt = render_prompt(messages)
    elif stage == "pretraining":
        if a.system:
            p.error("The base endpoint uses text completion; system-message behavior has not been trained")
        prompt = a.prompt
    else:
        p.error("This demo supports the retained pretraining and full-SFT checkpoints")
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids or len(ids)+a.max_new_tokens > model.config.max_position_embeddings:
        p.error("Prompt and requested response must fit the 1,024-token context")
    inputs = torch.tensor([ids], device=a.device)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False,
            max_new_tokens=a.max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(output[0, len(ids):], skip_special_tokens=True))


if __name__ == "__main__":
    main()
