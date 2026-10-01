"""
Captionz — Gradio UI (built for Hugging Face Spaces, also runs locally).

    python gradio_app.py [--backend ollama|hf] [--share]

Presentation layer only: every decision (prompt composition, backend, caption
policy, files) lives in captionz_core.py / captionz_hf.py. On Spaces the
backend defaults to "hf" (transformers, ZeroGPU); locally to "ollama".

Sources: upload files or a folder, or paste an image into the paste box.
Uploaded files are copied to a work folder; captions are saved alongside them
and can be downloaded as a ZIP archive.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import gradio as gr

from captionz_core import (
    BACKEND_LABELS, BACKENDS, CAPTION_LENGTHS, CAPTION_TYPES, DEFAULT_OLLAMA_URL, EXTRA_OPTIONS, IMAGE_EXTS,
    BatchProgress, Job, Settings, build_prompt, make_backend, run_jobs, save_pasted_image,
)
from captionz_llamacpp import DEFAULT_DIR as LC_DIR, DEFAULT_MODEL as LC_DEFAULT, KNOWN_MODELS as LC_KNOWN, \
    ModelRegistry, ServerBinary

ON_SPACES = bool(os.environ.get("SPACE_ID"))
DEFAULT_BACKEND = os.environ.get("CAPTIONZ_BACKEND", "hf" if ON_SPACES else "ollama")
WORK_DIR = Path(tempfile.gettempdir()) / "captionz_gradio"
_backend_cache: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Glue (thin): settings from widgets, backend cache, file handling
# --------------------------------------------------------------------------- #
def settings_from_ui(backend, url, model, hf_model, ctype, length, options, name, custom,
                     prefix, suffix, single, temperature, max_side, max_tokens=1024, no_think=True,
                     lc_model="") -> Settings:
    s = Settings.load()
    s.backend, s.ollama_url, s.model, s.hf_model = backend, url or DEFAULT_OLLAMA_URL, model or "", hf_model or ""
    s.llamacpp_model = lc_model or ""
    s.caption_type, s.caption_length, s.options = ctype, length, list(options or [])
    s.name, s.custom_prompt = name or "", custom or ""
    s.prefix, s.suffix, s.single_line = prefix or "", suffix or "", bool(single)
    s.temperature, s.max_side, s.max_tokens = float(temperature), int(max_side), int(max_tokens or 0)
    s.no_think = bool(no_think)
    s.existing, s.extension = "overwrite", ".txt"   # Spaces: temp copies, always overwrite
    return s.normalized()


def get_backend(s: Settings):
    key = f"{s.backend}|{s.ollama_url}|{s.hf_model}|{s.llamacpp_model}|{s.max_tokens}|{s.no_think}"
    if key not in _backend_cache:
        _backend_cache.clear()
        _backend_cache[key] = make_backend(s)
    return _backend_cache[key]


def lc_registry() -> ModelRegistry:
    s = Settings.load()
    return ModelRegistry(Path(s.llamacpp_dir) if s.llamacpp_dir else LC_DIR)


def list_models(backend, url, hf_model, lc_model):
    s = Settings.load()
    s.backend, s.ollama_url, s.hf_model, s.llamacpp_model = backend, url or DEFAULT_OLLAMA_URL, hf_model or "", lc_model or ""
    try:
        models = get_backend(s).list_models()
        status = f"✔ {len(models)} model(s)"
        if backend == "llamacpp" and not lc_registry().list():
            status = "The default model will be downloaded on first launch (about 2.9 GB)."
    except Exception as e:  # noqa: BLE001
        models, status = [], f"✖ {e}"
    upd = gr.update(choices=models, value=models[0] if models else None)
    return (upd if backend == "ollama" else gr.update(), upd if backend == "hf" else gr.update(),
            upd if backend == "llamacpp" else gr.update(), status)


def lc_action(kind, arg, progress=gr.Progress()):
    """Model management for the llama.cpp backend (runs in the request thread)."""
    reg = lc_registry()
    lines = []
    def lg(m):
        lines.append(m)
        progress(0, desc=m)
    try:
        if kind == "download":
            reg.add_known(arg or LC_DEFAULT["name"], lg)
        elif kind == "import":
            if not arg:
                raise ValueError("Select an Ollama model.")
            reg.import_from_ollama(arg, None, lg)
        elif kind == "update-server":
            ServerBinary(reg.models_dir.parent, Settings.load().llamacpp_build).update(lg)
        lines.append("✔ Finished")
    except Exception as e:  # noqa: BLE001
        lines.append(f"✖ {e}")
    models = reg.list() or [LC_DEFAULT["name"]]
    return gr.update(choices=models, value=models[0]), "\n".join(lines[-12:])


def preview_prompt(ctype, length, options, name, custom):
    return build_prompt(ctype, length, list(options or []), name or "", custom or "")


def import_files(files, items):
    """Copy uploaded files into the work folder and append them to the list."""
    items = list(items or [])
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    known = {it["path"] for it in items}
    added = 0
    for f in files or []:
        src = Path(f if isinstance(f, str) else getattr(f, "name", str(f)))
        if src.suffix.lower() not in IMAGE_EXTS:
            continue
        dst = WORK_DIR / src.name
        n = 1
        while dst.exists() and str(dst) not in known:
            dst = WORK_DIR / f"{src.stem}_{n}{src.suffix}"
            n += 1
        if str(dst) in known:
            continue
        shutil.copy(src, dst)
        items.append({"path": str(dst), "caption": "", "status": "pending", "seconds": 0.0})
        added += 1
    return items, *render(items), f"{added} image(s) added"


def import_pasted(img, items):
    if img is None:
        return items, *render(items), "No image pasted."
    items = list(items or [])
    out = save_pasted_image(img, WORK_DIR)
    items.append({"path": str(out), "caption": "", "status": "pending", "seconds": 0.0})
    return items, *render(items), f"Pasted image: {out.name}"


def clear_items():
    return [], *render([]), "List cleared"


def render(items):
    """Gallery + table views of the item list."""
    gallery = [(it["path"], Path(it["path"]).name) for it in items]
    table = [[Path(it["path"]).name, it["status"], f"{it['seconds']:.1f}s" if it["seconds"] else "",
              it["caption"]] for it in items]
    return gallery, table


def make_zip(items) -> str | None:
    done = [it for it in items if it["caption"]]
    if not done:
        return None
    zpath = WORK_DIR / f"captions_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for it in done:
            p = Path(it["path"])
            z.writestr(p.with_suffix(".txt").name, it["caption"] + "\n")
    return str(zpath)


def run_all(items, backend, url, model, hf_model, ctype, length, options, name, custom,
            prefix, suffix, single, temperature, max_side, max_tokens, no_think, lc_model, progress=gr.Progress()):
    items = list(items or [])
    if not items:
        yield items, *render(items), None, "Add images first."
        return
    s = settings_from_ui(backend, url, model, hf_model, ctype, length, options, name, custom,
                         prefix, suffix, single, temperature, max_side, max_tokens, no_think, lc_model)
    try:
        be = get_backend(s)
        if s.backend == "ollama" and not s.model:
            s.model = be.list_models()[0]
    except Exception as e:  # noqa: BLE001
        yield items, *render(items), None, f"Backend unavailable: {e}"
        return
    log = [f"Starting: {len(items)} image(s), backend {s.backend}, model {s.model or s.hf_model or s.llamacpp_model or 'default'}"]
    jobs = [Job(Path(it["path"])) for it in items]
    prog = BatchProgress()
    for ev in run_jobs(jobs, None, s, force=True, backend=be, progress=prog, log=log.append):
        if ev[0] == "progress":     # ("progress", done, total): nothing to render beyond the bar
            continue
        idx = ev[1]
        job, it = jobs[idx], items[idx]
        snap = prog.snapshot()
        progress(snap["fraction"], desc=snap["text"])
        if ev[0] == "phase" and ev[2] == "loading":
            log.append(f"Loading model “{prog.model}”…")
        if ev[0] == "row":
            it.update(status=job.status, caption=job.caption, seconds=job.duration)
            if job.status in ("ok", "error"):
                log.append(f"{'✔' if job.status == 'ok' else '✖'} {job.path.name} ({job.duration:.1f}s) "
                           f"{job.error or job.caption[:80]}")
        yield items, *render(items), None, "\n".join(log + [snap["text"]])
    ok = sum(it["status"] == "ok" for it in items)
    log.append(f"Finished: {ok}/{len(items)} successful · {prog.snapshot()['text']}")
    yield items, *render(items), make_zip(items), "\n".join(log)


def on_select(items, evt: gr.SelectData):
    idx = evt.index if isinstance(evt.index, int) else evt.index[0]
    if not items or idx is None or idx >= len(items):
        return idx, ""
    return idx, items[idx]["caption"]


def save_caption(items, idx, text):
    items = list(items or [])
    if idx is None or idx >= len(items):
        return items, *render(items), None, "Select an image in the gallery."
    items[idx]["caption"] = (text or "").strip()
    if items[idx]["status"] != "error":
        items[idx]["status"] = "ok"
    Path(items[idx]["path"]).with_suffix(".txt").write_text(items[idx]["caption"] + "\n", "utf-8")
    return items, *render(items), make_zip(items), f"Caption saved for {Path(items[idx]['path']).name}"


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def build(default_backend: str = DEFAULT_BACKEND) -> gr.Blocks:
    s0 = Settings.load()
    with gr.Blocks(title="Captionz") as demo:
        gr.Markdown("# Captionz\nBatch image captioning with vision models "
                    "(Ollama locally or `transformers` on Hugging Face Spaces).")
        items = gr.State([])
        sel_idx = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row():
                        backend = gr.Dropdown([(BACKEND_LABELS[b], b) for b in BACKENDS], value=default_backend,
                                              label="Backend", scale=1)
                        url = gr.Textbox(value=s0.ollama_url, label="Ollama URL", scale=2,
                                         visible=default_backend == "ollama")
                        model = gr.Dropdown([], value=None, label="Ollama model", scale=3,
                                            visible=default_backend == "ollama", allow_custom_value=True)
                        hf_model = gr.Dropdown([], value=None, label="Transformers model", scale=3,
                                               visible=default_backend == "hf", allow_custom_value=True)
                        lc_model = gr.Dropdown([], value=None, label="llama.cpp model (local)", scale=3,
                                               visible=default_backend == "llamacpp", allow_custom_value=True)
                        refresh = gr.Button("↻", scale=0, min_width=48)
                    status = gr.Markdown("…")
                    with gr.Accordion("llama.cpp models (no Ollama)", open=False, visible=default_backend == "llamacpp") as lc_box:
                        with gr.Row():
                            lc_known = gr.Dropdown(list(LC_KNOWN), value=LC_DEFAULT["name"], label="Download from Hugging Face")
                            lc_dl = gr.Button("Download")
                        with gr.Row():
                            try:
                                _oll = lc_registry().ollama_vision_models()
                            except Exception:
                                _oll = []
                            lc_oll = gr.Dropdown(_oll, value=_oll[0] if _oll else None, label="Import from Ollama",
                                                 allow_custom_value=True)
                            lc_imp = gr.Button("Import")
                            lc_srv = gr.Button("Update llama-server")
                        lc_log = gr.Textbox(lines=3, label="Model log", interactive=False)

                with gr.Group():
                    gr.Markdown("**Images**")
                    files = gr.File(file_count="multiple", type="filepath", label="Images (files or folder)",
                                    file_types=["image"], height=120)
                    with gr.Row():
                        paste = gr.Image(type="pil", sources=["clipboard", "upload"], label="Paste an image (Ctrl+V)",
                                         height=160)
                        with gr.Column():
                            add_paste = gr.Button("Add pasted image")
                            clear = gr.Button("Clear list", variant="secondary")

                with gr.Group():
                    gr.Markdown("**Prompt**")
                    with gr.Row():
                        ctype = gr.Dropdown(list(CAPTION_TYPES), value=s0.caption_type, label="Caption type", scale=2)
                        length = gr.Dropdown(list(CAPTION_LENGTHS), value=s0.caption_length, label="Length", scale=1)
                    with gr.Accordion("Additional options", open=False):
                        options = gr.CheckboxGroup(EXTRA_OPTIONS, value=[o for o in s0.options if o in EXTRA_OPTIONS],
                                                   label="", show_label=False)
                    name = gr.Textbox(value=s0.name, label="Character name ({name})",
                                      placeholder="blank = the main character")
                    custom = gr.Textbox(value=s0.custom_prompt, lines=2,
                                        label="Custom prompt (overrides type, length, and options when provided)")
                    final_prompt = gr.Textbox(lines=3, interactive=False, label="Final prompt sent to the model")

                with gr.Group():
                    gr.Markdown("**Output and model**")
                    with gr.Row():
                        prefix = gr.Textbox(value=s0.prefix, label="Prefix (trigger)")
                        suffix = gr.Textbox(value=s0.suffix, label="Suffix")
                        single = gr.Checkbox(value=s0.single_line, label="Single line")
                    with gr.Row():
                        temperature = gr.Slider(0, 1.5, value=s0.temperature, step=0.1, label="Temperature")
                        max_side = gr.Number(value=s0.max_side, precision=0, label="Maximum image side in px (0 = original)")
                        max_tokens = gr.Number(value=s0.max_tokens, precision=0, label="Maximum tokens (0 = unlimited)")
                        no_think = gr.Checkbox(value=s0.no_think, label="Disable reasoning")

                run = gr.Button("▶ Caption all", variant="primary")
                log = gr.Textbox(lines=6, label="Log", interactive=False)

            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Images", columns=3, height=320, allow_preview=True, type="filepath")
                table = gr.Dataframe(headers=["File", "Status", "Duration", "Caption"], type="array",
                                     interactive=False, wrap=True, label="Results")
                caption_box = gr.Textbox(lines=6, label="Selected image caption (editable)")
                save = gr.Button("💾 Save caption")
                zip_out = gr.File(label="Download captions (ZIP)", interactive=False)

        # ---- wiring ---------------------------------------------------------- #
        prompt_inputs = [ctype, length, options, name, custom]
        for w in prompt_inputs:
            w.change(preview_prompt, prompt_inputs, final_prompt)
        demo.load(preview_prompt, prompt_inputs, final_prompt)

        def on_backend(b):
            return (gr.update(visible=b == "ollama"), gr.update(visible=b == "ollama"), gr.update(visible=b == "hf"),
                    gr.update(visible=b == "llamacpp"), gr.update(visible=b == "llamacpp"))
        backend.change(on_backend, backend, [url, model, hf_model, lc_model, lc_box]) \
               .then(list_models, [backend, url, hf_model, lc_model], [model, hf_model, lc_model, status])
        refresh.click(list_models, [backend, url, hf_model, lc_model], [model, hf_model, lc_model, status])
        demo.load(list_models, [backend, url, hf_model, lc_model], [model, hf_model, lc_model, status])
        lc_dl.click(lambda k: lc_action("download", k), lc_known, [lc_model, lc_log])
        lc_imp.click(lambda o: lc_action("import", o), lc_oll, [lc_model, lc_log])
        lc_srv.click(lambda: lc_action("update-server", ""), None, [lc_model, lc_log])

        files.upload(import_files, [files, items], [items, gallery, table, log])
        add_paste.click(import_pasted, [paste, items], [items, gallery, table, log])
        clear.click(clear_items, None, [items, gallery, table, log])
        gallery.select(on_select, items, [sel_idx, caption_box])
        save.click(save_caption, [items, sel_idx, caption_box], [items, gallery, table, zip_out, log])
        run.click(run_all, [items, backend, url, model, hf_model, ctype, length, options, name, custom,
                            prefix, suffix, single, temperature, max_side, max_tokens, no_think, lc_model],
                  [items, gallery, table, zip_out, log])
    return demo


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    build(a.backend).launch(server_name=a.host, server_port=a.port, share=a.share, inbrowser=not a.no_browser)


if __name__ == "__main__":
    main()
