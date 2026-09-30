"""Verify a relocated candidate without accessing its original workspace."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path)
    args=parser.parse_args();root=args.directory.resolve()
    inventory=json.loads((root/"MANIFEST.json").read_text())["files"]
    def sha(path):
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
        return h.hexdigest()
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob("*") if p.is_file() and p.name!="MANIFEST.json"}
    if actual!=inventory:raise ValueError("Package inventory/checksums differ")
    record=json.loads((root/"package.json").read_text())
    if sha(root/"model/model.safetensors")!=record["selected_model_sha256"]:raise ValueError("Selected weights differ")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import torch
    model=AutoModelForCausalLM.from_pretrained(root/"model",local_files_only=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(root/"model",local_files_only=True)
    with torch.inference_mode():
        logits=model(**tokenizer("Test",return_tensors="pt")).logits
        if not torch.isfinite(logits).all():raise ValueError("Nonfinite offline forward")
    print(json.dumps({"verified":True,"weights_sha256":record["selected_model_sha256"],"offline_reload":True}))


if __name__=="__main__":main()
