"""
Captionz — benchmark Ollama vision models on an image or a folder.

Usage:
    python bench.py <image|folder> [--models a,b,c] [--exclude x,y] [--prompt "…"]
                    [--max-side 1024] [--url http://localhost:11434] [--out DIR]

Loops model → images: each model is loaded once (keep_alive during its run),
then unloaded before the next one. Ollama metrics per call (load, prompt eval,
generation, tok/s).

Output in --out (default: <folder>/captions_bench or <image>_bench):
    <image>__<model>.txt   one caption per image and per model (to compare)
    README.md              per-model table + every caption per image
    results.json           raw data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from captionz_core import DEFAULT_PROMPT, IMAGE_EXTS, OllamaClient  # noqa: E402

NS = 1e9


def short(model: str) -> str:
    """Nom de fichier sûr et court pour un modèle : 'qwen3-vl:8b' -> 'qwen3-vl_8b'."""
    s = model.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:60]


def call(client: OllamaClient, model: str, prompt: str, b64: str, keep_alive) -> dict:
    payload = {
        "model": model, "stream": False, "keep_alive": keep_alive,
        "options": {"temperature": 0.2},
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
    }
    t0 = time.time()
    try:
        r = client._request("POST", "/api/chat", payload)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}", "wall_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "wall_s": round(time.time() - t0, 1)}
    wall = time.time() - t0
    raw = (r.get("message") or {}).get("content", "")
    text = " ".join(OllamaClient.strip_thinking(raw).split())
    n, dur = r.get("eval_count", 0), r.get("eval_duration", 0) / NS
    return {
        "caption": text, "wall_s": round(wall, 1),
        "load_s": round(r.get("load_duration", 0) / NS, 1),
        "prompt_tokens": r.get("prompt_eval_count", 0),
        "prompt_eval_s": round(r.get("prompt_eval_duration", 0) / NS, 1),
        "gen_tokens": n, "gen_s": round(dur, 1),
        "tok_per_s": round(n / dur, 1) if dur else 0,
        "words": len(text.split()),
    }


def unload(client: OllamaClient, model: str) -> None:
    try:
        client._request("POST", "/api/generate", {"model": model, "keep_alive": 0})
    except Exception:
        pass


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # console Windows cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image ou dossier")
    ap.add_argument("--models", default="", help="liste explicite, séparée par des virgules")
    ap.add_argument("--exclude", default="", help="sous-chaînes de noms à exclure")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    src = Path(a.source)
    if src.is_dir():
        images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        out = Path(a.out) if a.out else src / "captions_bench"
    else:
        images = [src]
        out = Path(a.out) if a.out else src.with_name(src.stem + "_bench")
    out.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(a.url, timeout=900)
    models = [m for m in a.models.split(",") if m] or client.list_vision_models()
    excl = [e.lower() for e in a.exclude.split(",") if e]
    models = [m for m in models if not any(e in m.lower() for e in excl)]
    print(f"{len(images)} image(s) × {len(models)} modèle(s) → {out}\n")

    encoded = {img: OllamaClient.encode_image(img, a.max_side) for img in images}
    results: dict[str, dict[str, dict]] = {}   # model -> image name -> metrics
    t_start = time.time()
    for mi, model in enumerate(models, 1):
        print(f"[{mi}/{len(models)}] {model}", flush=True)
        results[model] = {}
        for ii, img in enumerate(images, 1):
            r = call(client, model, a.prompt, encoded[img], keep_alive="10m")
            results[model][img.name] = r
            if "error" in r:
                print(f"    {ii:>2}/{len(images)} {img.name}: ✖ {r['error']}", flush=True)
            else:
                (out / f"{img.stem}__{short(model)}.txt").write_text(r["caption"] + "\n", "utf-8")
                print(f"    {ii:>2}/{len(images)} {img.name}: {r['wall_s']}s "
                      f"(load {r['load_s']}s, gen {r['gen_s']}s, {r['tok_per_s']} tok/s, {r['words']} mots)", flush=True)
            (out / "results.json").write_text(json.dumps(
                {"source": str(src), "prompt": a.prompt, "max_side": a.max_side, "results": results},
                indent=2, ensure_ascii=False), "utf-8")
        unload(client, model)
        print()

    # --- rapport ---
    L = [f"# Bench Captionz — {src.name}", "",
         f"{len(images)} image(s), {len(models)} modèle(s), {time.strftime('%Y-%m-%d %H:%M')}, "
         f"durée totale {round((time.time() - t_start) / 60, 1)} min.", "",
         f"Prompt : `{a.prompt}`", "",
         f"Images réduites à {a.max_side} px de côté max, température 0.2. Chaque modèle est chargé une fois "
         f"(le chargement n'est compté que sur sa première image).", "",
         "Un fichier `<image>__<modèle>.txt` par caption est dans ce dossier.", "",
         "## Par modèle", "",
         "| Modèle | Moy. par image | Médiane | Min | Max | tok/s moy. | Mots moy. | Chargement | Erreurs |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for model, per in results.items():
        ok = [r for r in per.values() if "error" not in r]
        errs = len(per) - len(ok)
        if not ok:
            rows.append((9e9, f"| {model} | – | – | – | – | – | – | – | {errs} |"))
            continue
        # temps par image hors chargement (le chargement est payé une fois)
        times = sorted(r["wall_s"] - r["load_s"] for r in ok)
        avg = sum(times) / len(times)
        med = times[len(times) // 2]
        load = max(r["load_s"] for r in ok)
        tps = sum(r["tok_per_s"] for r in ok) / len(ok)
        words = sum(r["words"] for r in ok) / len(ok)
        rows.append((avg, f"| {model} | **{avg:.1f}s** | {med:.1f}s | {times[0]:.1f}s | {times[-1]:.1f}s "
                          f"| {tps:.0f} | {words:.0f} | {load:.1f}s | {errs} |"))
    L += [r for _, r in sorted(rows)]
    L += ["", "## Par image", ""]
    for img in images:
        L += [f"### {img.name}", ""]
        for model in models:
            r = results[model].get(img.name, {})
            head = f"**{model}** ({r.get('wall_s', '?')}s)"
            L += [head, "", r.get("caption") or f"*Erreur : {r.get('error')}*", ""]
    (out / "README.md").write_text("\n".join(L), "utf-8")
    print(f"→ {out / 'README.md'}")


if __name__ == "__main__":
    main()
