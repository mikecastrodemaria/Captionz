"""
Captionz — command-line interface (no GUI). Same core as the apps.

Examples:
    python cli.py photo.jpg
    python cli.py ./dataset --recursive --type "Booru tag list" --length short
    python cli.py ./dataset --model qwen3-vl:8b --option 3 --option 15 --name Lea --prefix "lea_style, "
    python cli.py ./dataset --backend hf --hf-model Qwen/Qwen2.5-VL-3B-Instruct
    python cli.py --list-models
    python cli.py --list-options
    python cli.py --show-prompt --type "Stable Diffusion prompt" --option 18

Options are referenced by their number in `--list-options`. Defaults come from
settings.json (the same file the GUIs use); flags override them for this run.
Use --save to persist the flags back to settings.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from captionz_core import (
    BACKENDS, CAPTION_LENGTHS, CAPTION_TYPES, EXTRA_OPTIONS, Job, Settings, collect_images, make_backend,
    run_jobs,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="captionz", description="Batch image captioning (Ollama or transformers)")
    ap.add_argument("paths", nargs="*", help="images and/or folders")
    ap.add_argument("-r", "--recursive", action="store_true", default=None, help="recurse into folders")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false")

    b = ap.add_argument_group("backend")
    b.add_argument("--backend", choices=BACKENDS, help="ollama (default) or hf (transformers)")
    b.add_argument("--url", help="Ollama URL (default http://localhost:11434)")
    b.add_argument("-m", "--model", help="Ollama model name")
    b.add_argument("--hf-model", help="transformers model id (backend hf)")
    b.add_argument("--keep-alive", help="Ollama keep_alive, e.g. 10m or 0")
    b.add_argument("--cpu", action="store_true", default=None, help="Ollama: force CPU (num_gpu=0)")
    b.add_argument("--temperature", type=float)
    b.add_argument("--max-tokens", type=int, help="cap generation (num_predict), default 1024, 0 = no cap")
    b.add_argument("--think", action="store_true", default=None,
                   help="allow thinking on models that support it (default: disabled, think=false)")
    b.add_argument("--no-think", dest="think", action="store_false", help="disable thinking (default)")
    b.add_argument("--max-side", type=int, help="downscale images to this max side before upload (0 = raw)")

    p = ap.add_argument_group("prompt")
    p.add_argument("-t", "--type", choices=list(CAPTION_TYPES), metavar="TYPE",
                   help="caption type: " + " | ".join(CAPTION_TYPES))
    p.add_argument("-l", "--length", choices=list(CAPTION_LENGTHS), metavar="LEN",
                   help="caption length: " + " | ".join(CAPTION_LENGTHS))
    p.add_argument("-o", "--option", action="append", type=int, metavar="N",
                   help="extra option number (see --list-options); repeatable. Replaces saved options.")
    p.add_argument("--no-options", action="store_true", help="use no extra option at all")
    p.add_argument("-n", "--name", help="character name substituted for {name}")
    p.add_argument("-p", "--prompt", help="custom prompt (overrides type/length/options)")

    o = ap.add_argument_group("output")
    o.add_argument("--prefix")
    o.add_argument("--suffix")
    o.add_argument("--ext", help="caption file extension (default .txt)")
    o.add_argument("--existing", choices=["skip", "overwrite", "append"], help="what to do with existing captions")
    o.add_argument("--multiline", action="store_true", help="keep line breaks (default: single line)")
    o.add_argument("--json", metavar="FILE", help="also write a JSON report {file: {status, caption, seconds}}")
    o.add_argument("-q", "--quiet", action="store_true", help="only print errors and the summary")

    i = ap.add_argument_group("info")
    i.add_argument("--list-models", action="store_true", help="list vision models of the backend and exit")
    i.add_argument("--list-options", action="store_true", help="list extra options with their numbers and exit")
    i.add_argument("--show-prompt", action="store_true", help="print the final prompt and exit")
    i.add_argument("--save", action="store_true", help="persist the given flags into settings.json")
    return ap


def settings_from_args(a: argparse.Namespace) -> Settings:
    s = Settings.load()
    if a.backend:
        s.backend = a.backend
    if a.url:
        s.ollama_url = a.url
    if a.model:
        s.model = a.model
    if a.hf_model:
        s.hf_model = a.hf_model
    if a.keep_alive is not None:
        s.keep_alive = a.keep_alive
    if a.cpu is not None:
        s.cpu_only = a.cpu
    if a.temperature is not None:
        s.temperature = a.temperature
    if a.max_tokens is not None:
        s.max_tokens = a.max_tokens
    if a.think is not None:
        s.no_think = not a.think
    if a.max_side is not None:
        s.max_side = a.max_side
    if a.type:
        s.caption_type = a.type
    if a.length:
        s.caption_length = a.length
    if a.no_options:
        s.options = []
    elif a.option:
        bad = [n for n in a.option if not 1 <= n <= len(EXTRA_OPTIONS)]
        if bad:
            sys.exit(f"unknown option number(s): {bad} (see --list-options)")
        s.options = [EXTRA_OPTIONS[n - 1] for n in a.option]
    if a.name is not None:
        s.name = a.name
    if a.prompt is not None:
        s.custom_prompt = a.prompt
    if a.prefix is not None:
        s.prefix = a.prefix
    if a.suffix is not None:
        s.suffix = a.suffix
    if a.ext:
        s.extension = a.ext
    if a.existing:
        s.existing = a.existing
    if a.multiline:
        s.single_line = False
    if a.recursive is not None:
        s.recursive = a.recursive
    return s.normalized()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows console in cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = build_parser()
    a = ap.parse_args(argv)
    s = settings_from_args(a)

    if a.list_options:
        for n, opt in enumerate(EXTRA_OPTIONS, 1):
            mark = "*" if opt in s.options else " "
            print(f"{n:>2} {mark} {opt}")
        print("\n* = enabled in settings.json")
        return 0
    if a.show_prompt:
        print(s.prompt)
        return 0
    if a.list_models:
        try:
            models = make_backend(s).list_models()
        except Exception as e:  # noqa: BLE001
            sys.exit(f"cannot list models: {e}")
        for m in models:
            print(("* " if m == (s.model if s.backend == "ollama" else s.hf_model) else "  ") + m)
        return 0
    if a.save:
        s.save()
        print(f"settings saved to settings.json")
        if not a.paths:
            return 0
    if not a.paths:
        ap.print_help()
        return 1

    paths = [Path(p) for p in a.paths]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit("not found: " + ", ".join(map(str, missing)))
    images = collect_images(paths, s.recursive)
    if not images:
        sys.exit("no image found")

    backend = make_backend(s)
    if s.backend == "ollama" and not s.model:
        try:
            models = backend.list_models()
        except Exception as e:  # noqa: BLE001
            sys.exit(f"cannot reach Ollama at {s.ollama_url}: {e}")
        if not models:
            sys.exit("no vision model installed (e.g. ollama pull qwen3-vl:8b)")
        s.model = models[0]
    model_label = s.model if s.backend == "ollama" else (s.hf_model or "default")
    if not a.quiet:
        print(f"{len(images)} image(s) · backend {s.backend} · model {model_label}")
        print(f"prompt: {s.prompt}\n")

    jobs = [Job(p) for p in images]
    stop = threading.Event()
    t0 = time.time()
    try:
        for ev in run_jobs(jobs, None, s, stop_event=stop, backend=backend):
            if ev[0] == "row":
                job = jobs[ev[1]]
                if job.status == "en cours":
                    continue
                if job.status == "erreur":
                    print(f"✖ {job.path}: {job.error}", file=sys.stderr)
                elif not a.quiet:
                    tag = "·" if job.status == "ignoré" else "✔"
                    detail = job.error if job.status == "ignoré" else job.caption[:100] + ("…" if len(job.caption) > 100 else "")
                    print(f"{tag} {job.path.name} ({job.duration:.1f}s) {detail}")
    except KeyboardInterrupt:
        stop.set()
        print("\ninterrupted", file=sys.stderr)

    ok = sum(j.status == "ok" for j in jobs)
    err = sum(j.status == "erreur" for j in jobs)
    skip = sum(j.status == "ignoré" for j in jobs)
    print(f"\n{ok} ok, {skip} skipped, {err} error(s) in {time.time() - t0:.1f}s")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {str(j.path): {"status": j.status, "caption": j.caption, "error": j.error,
                           "seconds": round(j.duration, 1)} for j in jobs},
            indent=2, ensure_ascii=False), "utf-8")
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
