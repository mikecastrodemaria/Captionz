"""
Push the Gradio version to a Hugging Face Space.

    pip install huggingface_hub
    huggingface-cli login
    python deploy_space.py <user>/<space-name> [--create] [--private]

Uploads only the files the Space needs (core, HF backend, Gradio UI, Space
entry point, requirements, README). Local UIs and benchmarks stay out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
FILES = {  # local path -> path in the Space
    "captionz_core.py": "captionz_core.py",
    "captionz_hf.py": "captionz_hf.py",
    "gradio_app.py": "gradio_app.py",
    "hf_space/app.py": "app.py",
    "hf_space/requirements.txt": "requirements.txt",
    "hf_space/README.md": "README.md",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id", help="<user>/<space-name>")
    ap.add_argument("--create", action="store_true", help="create the Space if it does not exist")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--hardware", default="zero-a10g", help="Space hardware when creating (default zero-a10g)")
    a = ap.parse_args()
    api = HfApi()
    if a.create:
        api.create_repo(a.repo_id, repo_type="space", space_sdk="gradio", private=a.private,
                        space_hardware=a.hardware, exist_ok=True)
    for local, remote in FILES.items():
        api.upload_file(path_or_fileobj=str(ROOT / local), path_in_repo=remote, repo_id=a.repo_id, repo_type="space")
        print(f"uploaded {local} -> {remote}")
    print(f"done: https://huggingface.co/spaces/{a.repo_id}")


if __name__ == "__main__":
    main()
