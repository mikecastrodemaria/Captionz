"""
Captionz — Gradio UI (built for Hugging Face Spaces, also runs locally).

    python gradio_app.py [--backend ollama|hf] [--share]

Presentation layer only: every decision (prompt composition, backend, caption
policy, files) lives in captionz_core.py / captionz_hf.py. On Spaces the
backend defaults to "hf" (transformers, ZeroGPU); locally to "ollama".

Sources: upload (files or a folder), or paste an image into the paste box.
Uploaded files are copied to a work folder; captions are written next to them
and offered as a zip.
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
    BACKENDS, CAPTION_LENGTHS, CAPTION_TYPES, DEFAULT_OLLAMA_URL, EXTRA_OPTIONS, IMAGE_EXTS, Job, Settings,
    build_prompt, caption_job, make_backend, save_pasted_image,
)

ON_SPACES = bool(os.environ.get("SPACE_ID"))
DEFAULT_BACKEND = os.environ.get("CAPTIONZ_BACKEND", "hf" if ON_SPACES else "ollama")
WORK_DIR = Path(tempfile.gettempdir()) / "captionz_gradio"
_backend_cache: dict[str, object] = {}


# --------------------------------------------------------------------------- #
# Glue (thin): settings from widgets, backend cache, file handling
# --------------------------------------------------------------------------- #
def settings_from_ui(backend, url, model, hf_model, ctype, length, options, name, custom,
                     prefix, suffix, single, temperature, max_side) -> Settings:
    s = Settings.load()
    s.backend, s.ollama_url, s.model, s.hf_model = backend, url or DEFAULT_OLLAMA_URL, model or "", hf_model or ""
    s.caption_type, s.caption_length, s.options = ctype, length, list(options or [])
    s.name, s.custom_prompt = name or "", custom or ""
    s.prefix, s.suffix, s.single_line = prefix or "", suffix or "", bool(single)
    s.temperature, s.max_side = float(temperature), int(max_side)
    s.existing, s.extension = "overwrite", ".txt"   # Spaces: temp copies, always overwrite
    return s.normalized()


def get_backend(s: Settings):
    key = f"{s.backend}|{s.ollama_url}|{s.hf_model}"
    if key not in _backend_cache:
        _backend_cache.clear()
        _backend_cache[key] = make_backend(s)
    return _backend_cache[key]


def list_models(backend, url, hf_model):
    s = Settings.load()
    s.backend, s.ollama_url, s.hf_model = backend, url or DEFAULT_OLLAMA_URL, hf_model or ""
    try:
        models = get_backend(s).list_models()
        status = f"✔ {len(models)} modèle(s)"
    except Exception as e:  # noqa: BLE001
        models, status = [], f"✖ {e}"
    if backend == "ollama":
        return gr.update(choices=models, value=models[0] if models else None), gr.update(), status
    return gr.update(), gr.update(choices=models, value=models[0] if models else None), status


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
        items.append({"path": str(dst), "caption": "", "status": "en attente", "seconds": 0.0})
        added += 1
    return items, *render(items), f"{added} image(s) ajoutée(s)"


def import_pasted(img, items):
    if img is None:
        return items, *render(items), "Aucune image collée"
    items = list(items or [])
    out = save_pasted_image(img, WORK_DIR)
    items.append({"path": str(out), "caption": "", "status": "en attente", "seconds": 0.0})
    return items, *render(items), f"Image collée : {out.name}"


def clear_items():
    return [], *render([]), "Liste vidée"


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
            prefix, suffix, single, temperature, max_side, progress=gr.Progress()):
    items = list(items or [])
    if not items:
        yield items, *render(items), None, "Ajoute d'abord des images."
        return
    s = settings_from_ui(backend, url, model, hf_model, ctype, length, options, name, custom,
                         prefix, suffix, single, temperature, max_side)
    try:
        be = get_backend(s)
        if s.backend == "ollama" and not s.model:
            s.model = be.list_models()[0]
    except Exception as e:  # noqa: BLE001
        yield items, *render(items), None, f"Backend indisponible : {e}"
        return
    log = [f"Démarrage : {len(items)} image(s), backend {s.backend}, modèle {s.model or s.hf_model or 'défaut'}"]
    for i, it in enumerate(progress.tqdm(items, desc="captioning")):
        job = Job(Path(it["path"]))
        it["status"] = "en cours"
        yield items, *render(items), None, "\n".join(log)
        caption_job(job, s, be, force=True)
        it.update(status=job.status, caption=job.caption, seconds=job.duration)
        log.append(f"{'✔' if job.status == 'ok' else '✖'} {job.path.name} ({job.duration:.1f}s) "
                   f"{job.error or job.caption[:80]}")
        yield items, *render(items), None, "\n".join(log)
    ok = sum(it["status"] == "ok" for it in items)
    log.append(f"Terminé : {ok}/{len(items)} ok")
    yield items, *render(items), make_zip(items), "\n".join(log)


def on_select(items, evt: gr.SelectData):
    idx = evt.index if isinstance(evt.index, int) else evt.index[0]
    if not items or idx is None or idx >= len(items):
        return idx, ""
    return idx, items[idx]["caption"]


def save_caption(items, idx, text):
    items = list(items or [])
    if idx is None or idx >= len(items):
        return items, *render(items), None, "Sélectionne une image dans la galerie."
    items[idx]["caption"] = (text or "").strip()
    if items[idx]["status"] != "erreur":
        items[idx]["status"] = "ok"
    Path(items[idx]["path"]).with_suffix(".txt").write_text(items[idx]["caption"] + "\n", "utf-8")
    return items, *render(items), make_zip(items), f"Caption enregistrée pour {Path(items[idx]['path']).name}"


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def build(default_backend: str = DEFAULT_BACKEND) -> gr.Blocks:
    s0 = Settings.load()
    with gr.Blocks(title="Captionz") as demo:
        gr.Markdown("# Captionz\nBatch image captioning with vision models "
                    "(Ollama locally, `transformers` on Hugging Face Spaces).")
        items = gr.State([])
        sel_idx = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row():
                        backend = gr.Dropdown(list(BACKENDS), value=default_backend, label="Backend", scale=1)
                        url = gr.Textbox(value=s0.ollama_url, label="Ollama URL", scale=2,
                                         visible=default_backend == "ollama")
                        model = gr.Dropdown([], value=None, label="Modèle Ollama", scale=3,
                                            visible=default_backend == "ollama", allow_custom_value=True)
                        hf_model = gr.Dropdown([], value=None, label="Modèle transformers", scale=3,
                                               visible=default_backend == "hf", allow_custom_value=True)
                        refresh = gr.Button("↻", scale=0, min_width=48)
                    status = gr.Markdown("…")

                with gr.Group():
                    gr.Markdown("**Sources**")
                    files = gr.File(file_count="multiple", type="filepath", label="Images (fichiers ou dossier)",
                                    file_types=["image"], height=120)
                    with gr.Row():
                        paste = gr.Image(type="pil", sources=["clipboard", "upload"], label="Coller une image (Ctrl+V)",
                                         height=160)
                        with gr.Column():
                            add_paste = gr.Button("Ajouter l'image collée")
                            clear = gr.Button("Vider la liste", variant="secondary")

                with gr.Group():
                    gr.Markdown("**Prompt**")
                    with gr.Row():
                        ctype = gr.Dropdown(list(CAPTION_TYPES), value=s0.caption_type, label="Type de caption", scale=2)
                        length = gr.Dropdown(list(CAPTION_LENGTHS), value=s0.caption_length, label="Longueur", scale=1)
                    with gr.Accordion("Options supplémentaires", open=False):
                        options = gr.CheckboxGroup(EXTRA_OPTIONS, value=[o for o in s0.options if o in EXTRA_OPTIONS],
                                                   label="", show_label=False)
                    name = gr.Textbox(value=s0.name, label="Nom du personnage ({name})",
                                      placeholder="vide = the main character")
                    custom = gr.Textbox(value=s0.custom_prompt, lines=2,
                                        label="Prompt personnalisé (remplace type / longueur / options si rempli)")
                    final_prompt = gr.Textbox(lines=3, interactive=False, label="Prompt final envoyé au modèle")

                with gr.Group():
                    gr.Markdown("**Sortie et modèle**")
                    with gr.Row():
                        prefix = gr.Textbox(value=s0.prefix, label="Préfixe (trigger)")
                        suffix = gr.Textbox(value=s0.suffix, label="Suffixe")
                        single = gr.Checkbox(value=s0.single_line, label="Une seule ligne")
                    with gr.Row():
                        temperature = gr.Slider(0, 1.5, value=s0.temperature, step=0.1, label="Température")
                        max_side = gr.Number(value=s0.max_side, precision=0, label="Côté max px (0 = brut)")

                run = gr.Button("▶ Captionner tout", variant="primary")
                log = gr.Textbox(lines=6, label="Journal", interactive=False)

            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Images", columns=3, height=320, allow_preview=True, type="filepath")
                table = gr.Dataframe(headers=["Fichier", "Statut", "Durée", "Caption"], type="array",
                                     interactive=False, wrap=True, label="Résultats")
                caption_box = gr.Textbox(lines=6, label="Caption de l'image sélectionnée (éditable)")
                save = gr.Button("💾 Enregistrer la caption")
                zip_out = gr.File(label="Télécharger les captions (zip)", interactive=False)

        # ---- wiring ---------------------------------------------------------- #
        prompt_inputs = [ctype, length, options, name, custom]
        for w in prompt_inputs:
            w.change(preview_prompt, prompt_inputs, final_prompt)
        demo.load(preview_prompt, prompt_inputs, final_prompt)

        def on_backend(b):
            return (gr.update(visible=b == "ollama"), gr.update(visible=b == "ollama"), gr.update(visible=b == "hf"))
        backend.change(on_backend, backend, [url, model, hf_model]) \
               .then(list_models, [backend, url, hf_model], [model, hf_model, status])
        refresh.click(list_models, [backend, url, hf_model], [model, hf_model, status])
        demo.load(list_models, [backend, url, hf_model], [model, hf_model, status])

        files.upload(import_files, [files, items], [items, gallery, table, log])
        add_paste.click(import_pasted, [paste, items], [items, gallery, table, log])
        clear.click(clear_items, None, [items, gallery, table, log])
        gallery.select(on_select, items, [sel_idx, caption_box])
        save.click(save_caption, [items, sel_idx, caption_box], [items, gallery, table, zip_out, log])
        run.click(run_all, [items, backend, url, model, hf_model, ctype, length, options, name, custom,
                            prefix, suffix, single, temperature, max_side],
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
