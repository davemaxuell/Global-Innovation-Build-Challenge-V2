"""Upload a prepared release folder (release/hf_*) to the Hugging Face Hub, then verify it.

The token is taken from the HF_TOKEN environment variable or typed at a hidden prompt.
It is passed to the API directly and never written to disk: this server account is
shared, so ``hf auth login`` (which saves the token) must not be used. A fine-grained
token with write access to the target repository is enough; revoke it afterwards.

The folder's RELEASE_MANIFEST.json must list every file with its SHA256; the upload is
refused if anything differs. After the upload, each remote file's size and (for LFS
files) SHA256 are checked against the manifest.

Run: python scripts/upload_release.py --folder release/hf_v2_1 --repo <user>/scglm-v2.1-46m --public
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--repo", required=True, help="<namespace>/<name>")
    vis = p.add_mutually_exclusive_group(required=True)
    vis.add_argument("--public", action="store_true")
    vis.add_argument("--private", action="store_true")
    p.add_argument("--message", default="Upload model release")
    args = p.parse_args()

    from huggingface_hub import HfApi
    manifest = json.loads((args.folder / "RELEASE_MANIFEST.json").read_text())["files_sha256"]
    local = {f.name for f in args.folder.iterdir() if f.is_file() and f.name != "RELEASE_MANIFEST.json"}
    if local != set(manifest):
        raise SystemExit(f"Folder files differ from the manifest: {sorted(local ^ set(manifest))}")
    bad = [n for n, h in manifest.items() if sha256(args.folder / n) != h]
    if bad:
        raise SystemExit(f"Hash mismatch against RELEASE_MANIFEST.json: {bad}")
    if any(t in (args.folder / "README.md").read_text() for t in ("<hf-username>", "<github-repo-url>", "<authors>")):
        raise SystemExit("README.md still contains placeholders")

    token = os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face write token (hidden, not saved): ")
    api = HfApi(token=token)
    who = api.whoami()
    user, orgs = who["name"], [o["name"] for o in who.get("orgs", [])]
    if args.repo.split("/")[0] not in [user] + orgs:
        raise SystemExit(f"This token belongs to '{user}' (orgs: {orgs}); it cannot create or write {args.repo}")
    print(f"Authenticated as {user}; uploading {args.folder} to {args.repo} ({'public' if args.public else 'private'})")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    # create_repo(exist_ok=True) leaves an existing repo's visibility unchanged; set it explicitly.
    api.update_repo_settings(args.repo, private=args.private, repo_type="model")
    commit = api.upload_folder(folder_path=str(args.folder), repo_id=args.repo, repo_type="model",
                               commit_message=args.message)
    print("Commit:", commit.commit_url if hasattr(commit, "commit_url") else commit)

    info = api.model_info(args.repo, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings}
    problems = []
    for name in sorted(local | {"RELEASE_MANIFEST.json"}):
        s = remote.get(name)
        if s is None:
            problems.append(f"{name}: missing on the Hub"); continue
        if s.size != (args.folder / name).stat().st_size:
            problems.append(f"{name}: size {s.size} != local")
        if s.lfs is not None and name in manifest and s.lfs.sha256 != manifest[name]:
            problems.append(f"{name}: LFS sha256 differs")
    print(json.dumps({"repo": f"https://huggingface.co/{args.repo}", "private": info.private,
                      "files": len(remote), "verified": not problems, "problems": problems}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
