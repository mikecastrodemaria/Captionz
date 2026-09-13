"""
Captionz — manage llama.cpp models and the llama-server binary (no Ollama needed).

    python captionz_models.py list                       # local models + server build
    python captionz_models.py download                   # default model (Qwen2.5-VL-3B Q4_K_M)
    python captionz_models.py download gemma-3-4b-it-Q4_K_M
    python captionz_models.py add NAME --repo ORG/REPO --model FILE.gguf --mmproj MMPROJ.gguf
    python captionz_models.py ollama                     # importable Ollama models (with a vision projector)
    python captionz_models.py import "nutboy02/Agents-A1-4B-Kimi-heretic:latest" [--name kimi-4b]
    python captionz_models.py update [NAME]              # re-download HF models whose files changed
    python captionz_models.py server                     # install / show llama-server
    python captionz_models.py server --update            # newest llama.cpp release
    python captionz_models.py remove NAME
    python captionz_models.py test [NAME] image.jpg      # quick caption through llama-server

Models live in <app>/llamacpp/models/<name>/ (settings.llamacpp_dir).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from captionz_core import Settings
from captionz_llamacpp import DEFAULT_MODEL, KNOWN_MODELS, LlamaCppBackend, ModelRegistry, ServerBinary, pick_build


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="captionz_models", description="llama.cpp models for Captionz")
    ap.add_argument("--dir", help="llama.cpp folder (default: settings.llamacpp_dir or <app>/llamacpp)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("download"); d.add_argument("name", nargs="?", default=DEFAULT_MODEL["name"],
                                                    choices=list(KNOWN_MODELS))
    a = sub.add_parser("add"); a.add_argument("name"); a.add_argument("--repo", required=True)
    a.add_argument("--model", required=True); a.add_argument("--mmproj", required=True)
    sub.add_parser("ollama")
    i = sub.add_parser("import"); i.add_argument("ollama_name"); i.add_argument("--name")
    u = sub.add_parser("update"); u.add_argument("name", nargs="?")
    s = sub.add_parser("server"); s.add_argument("--update", action="store_true")
    r = sub.add_parser("remove"); r.add_argument("name")
    t = sub.add_parser("test"); t.add_argument("name", nargs="?"); t.add_argument("image")
    args = ap.parse_args(argv)

    st = Settings.load()
    base = Path(args.dir or st.llamacpp_dir or (Path(__file__).resolve().parent / "llamacpp"))
    reg = ModelRegistry(base)
    binary = ServerBinary(base, st.llamacpp_build)
    log = print

    if args.cmd == "list":
        cur = binary.current()
        print(f"folder : {base}")
        print(f"server : {cur['tag'] + ' (' + cur['build'] + ')' if cur else 'not installed (auto on first use, build ' + pick_build(st.llamacpp_build) + ')'}")
        models = reg.list()
        print("models :" if models else "models : none (run `download` or `import`)")
        for m in models:
            info = reg.info(m)
            mp, pp = reg.paths(m)
            size = (mp.stat().st_size + pp.stat().st_size) >> 20
            mark = "*" if m == st.llamacpp_model else " "
            print(f"  {mark} {m}  [{info.get('source', '?')}] {size} Mo  {info.get('repo') or info.get('ollama_name') or ''}")
        print("\nknown (download): " + ", ".join(KNOWN_MODELS))
        return 0
    if args.cmd == "download":
        reg.add_known(args.name, log); return 0
    if args.cmd == "add":
        reg.add_from_hf(args.name, args.repo, args.model, args.mmproj, log); return 0
    if args.cmd == "ollama":
        for m in reg.ollama_vision_models():
            print(("* " if reg.has(m.split('/')[-1].replace(':', '_')) else "  ") + m)
        return 0
    if args.cmd == "import":
        reg.import_from_ollama(args.ollama_name, args.name, log); return 0
    if args.cmd == "update":
        names = [args.name] if args.name else reg.list()
        for n in names:
            reg.update_from_source(n, log)
        return 0
    if args.cmd == "server":
        if args.update:
            binary.update(log)
        else:
            print(binary.ensure(log))
        return 0
    if args.cmd == "remove":
        reg.remove(args.name); print(f"removed {args.name}"); return 0
    if args.cmd == "test":
        be = LlamaCppBackend(base, args.name or st.llamacpp_model or DEFAULT_MODEL["name"], st.llamacpp_build)
        be.on_progress = log
        import time
        t0 = time.time()
        print(be.caption(be.model, "Describe this image in one short sentence.", Path(args.image)))
        print(f"({time.time() - t0:.1f}s)")
        LlamaCppBackend.shutdown()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
