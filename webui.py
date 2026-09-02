"""
Captionz — NiceGUI web UI (same core as the Tkinter app).

    python app.py --ui web [--port 8080] [--host 0.0.0.0] [--no-browser]
    python webui.py            # direct launch

Single-user local tool: state lives at module level (settings, jobs, captioner).
Sources: a local path (file or folder), browser upload, or paste (Ctrl+V) of an
image or screenshot into the page.
"""

from __future__ import annotations

import asyncio
import base64
import io
import queue
import time
from pathlib import Path

from nicegui import run, ui

from captionz_core import (
    APP_TITLE, CAPTION_LENGTHS, CAPTION_TYPES, DEFAULT_OLLAMA_URL, EXTRA_OPTIONS, IMAGE_EXTS,
    Captioner, Job, OllamaClient, Settings, collect_images, save_pasted_image,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

STATUS_COLOR = {"ok": "text-green-7", "erreur": "text-red-7", "ignoré": "text-amber-8", "en cours": "text-blue-7"}

# ---- module state (single user) -------------------------------------------- #
settings = Settings.load()
jobs: list[Job] = []
captioner = Captioner()
W: dict[str, object] = {}   # widgets


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    W["log"].push(time.strftime("[%H:%M:%S] ") + msg)


def collect_settings() -> Settings:
    s = Settings(
        ollama_url=(W["url"].value or "").strip() or DEFAULT_OLLAMA_URL,
        model=W["model"].value or "",
        caption_type=W["type"].value,
        caption_length=W["length"].value,
        options=[o for o, cb in W["opts"] if cb.value],
        name=W["name"].value or "",
        custom_prompt=(W["custom"].value or "").strip(),
        prefix=W["prefix"].value or "",
        suffix=W["suffix"].value or "",
        extension=(W["ext"].value or "").strip() or ".txt",
        recursive=W["recursive"].value,
        existing=W["existing"].value,
        temperature=float(W["temp"].value or 0),
        max_tokens=int(W["maxtok"].value or 0),
        single_line=W["single"].value,
        keep_alive=str(W["keep"].value or "0"),
        max_side=int(W["maxside"].value or 0),
        cpu_only=W["cpu"].value,
        dark=bool(W["dark"].value),
        paste_dir=settings.paste_dir,
        vision_blocklist=list(settings.vision_blocklist),
    ).normalized()
    return s


def update_prompt_preview(*_) -> None:
    try:
        W["preview"].value = collect_settings().prompt
    except Exception:
        pass


def row_of(idx: int) -> dict:
    j = jobs[idx]
    if j.status == "en cours" and j.started:
        t = f"{time.time() - j.started:.0f}s…"
    else:
        t = f"{j.duration:.1f}s" if j.duration else ""
    return {"id": idx, "file": str(j.path), "name": j.path.name, "status": j.status, "time": t}


def refresh_table(keep_selection: bool = True) -> None:
    t = W["table"]
    sel = {r["id"] for r in t.selected} if keep_selection else set()
    t.rows = [row_of(i) for i in range(len(jobs))]
    t.selected = [r for r in t.rows if r["id"] in sel]
    t.update()
    W["count"].text = f"{len(jobs)} image{'s' if len(jobs) > 1 else ''}"


def selected_indices() -> list[int]:
    return sorted(r["id"] for r in W["table"].selected)


def current_index() -> int | None:
    sel = selected_indices()
    return sel[0] if sel else None


def thumbnail_data_url(path: Path, max_side: int = 900) -> str | None:
    if Image is None:
        return None
    try:
        img = Image.open(path)
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def show_selected() -> None:
    idx = current_index()
    if idx is None:
        W["image"].set_source("")
        W["caption"].value = ""
        W["capfile"].text = ""
        return
    j = jobs[idx]
    W["image"].set_source(thumbnail_data_url(j.path) or "")
    cap = j.path.with_suffix(collect_settings().extension)
    text = j.caption
    if not text and cap.exists():
        try:
            text = cap.read_text("utf-8").strip()
        except Exception:
            text = ""
    W["caption"].value = text
    W["capfile"].text = cap.name + (" (existe)" if cap.exists() else "")


def add_paths(paths: list[Path]) -> None:
    images = collect_images(paths, W["recursive"].value)
    existing = {j.path for j in jobs}
    added = 0
    for img in images:
        if img not in existing:
            jobs.append(Job(img))
            added += 1
    refresh_table()
    log(f"{added} image(s) ajoutée(s) ({len(images) - added} doublon(s) ignoré(s)).")
    if added:
        W["table"].selected = [W["table"].rows[-1]]
        W["table"].update()
        show_selected()


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
async def refresh_models() -> None:
    url = (W["url"].value or "").strip() or DEFAULT_OLLAMA_URL
    W["conn"].text = "connexion…"
    try:
        models = await run.io_bound(OllamaClient(url, timeout=15).list_vision_models, settings.vision_blocklist)
    except Exception as e:  # noqa: BLE001
        W["conn"].text = "✖ hors ligne"
        W["model"].options = []
        W["model"].update()
        log(f"Impossible de joindre Ollama : {e}")
        return
    W["model"].options = models
    if models:
        if W["model"].value not in models:
            W["model"].value = models[0]
        W["conn"].text = f"✔ {len(models)} modèle(s) vision"
    else:
        W["conn"].text = "aucun modèle vision"
        log("Aucun modèle vision trouvé. Exemple : ollama pull qwen3-vl:8b")
    W["model"].update()


def add_path_from_input() -> None:
    raw = (W["path"].value or "").strip().strip('"')
    if not raw:
        ui.notify("Indique un chemin de fichier ou de dossier.", type="warning")
        return
    p = Path(raw)
    if not p.exists():
        ui.notify(f"Introuvable : {p}", type="negative")
        return
    add_paths([p])
    W["path"].value = ""


async def on_upload(e) -> None:
    data = await e.file.read()
    out_dir = settings.paste_path / "uploads"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(e.file.name).name
    out = out_dir / name
    n = 1
    while out.exists():
        out = out_dir / f"{Path(name).stem}_{n}{Path(name).suffix}"
        n += 1
    out.write_bytes(data)
    if out.suffix.lower() in IMAGE_EXTS:
        add_paths([out])
    else:
        log(f"Ignoré (pas une image) : {name}")


def on_paste(e) -> None:
    """Image pasted into the page (JS listener below sends a data URL)."""
    args = e.args if isinstance(e.args, dict) else (e.args[0] if e.args else {})
    data_url = args.get("data", "")
    if not data_url.startswith("data:image") or Image is None:
        log("Aucune image dans le presse-papiers.")
        return
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw))
    out = save_pasted_image(img, settings.paste_path)
    log(f"Image collée enregistrée : {out}")
    add_paths([out])


def remove_selected() -> None:
    if captioner.is_running():
        return
    sel = set(selected_indices())
    if not sel:
        return
    jobs[:] = [j for i, j in enumerate(jobs) if i not in sel]
    refresh_table(keep_selection=False)
    show_selected()


def clear_jobs() -> None:
    if captioner.is_running():
        return
    jobs.clear()
    refresh_table(keep_selection=False)
    show_selected()


def save_caption() -> None:
    idx = current_index()
    if idx is None:
        ui.notify("Sélectionne une image dans la liste.", type="warning")
        return
    j = jobs[idx]
    text = (W["caption"].value or "").strip()
    out = j.path.with_suffix(collect_settings().extension)
    out.write_text(text + "\n", encoding="utf-8")
    j.caption = text
    if j.status != "erreur":
        j.status = "ok"
    refresh_table()
    W["capfile"].text = out.name + " (existe)"
    log(f"💾 {out.name} enregistré.")
    ui.notify("Caption enregistrée", type="positive")


def start(indices: list[int] | None, force: bool = False) -> None:
    global settings
    if captioner.is_running():
        return
    s = collect_settings()
    if not s.model:
        ui.notify("Sélectionne un modèle vision.", type="warning")
        return
    if not jobs:
        ui.notify("Ajoute au moins une image ou un dossier.", type="warning")
        return
    if indices is None:
        indices = list(range(len(jobs)))
    settings = s
    s.save()
    W["btn_all"].disable()
    W["btn_sel"].disable()
    W["btn_stop"].enable()
    W["progress"].value = 0
    W["progress_label"].text = "0%"
    W["status"].text = "démarrage…"
    log(f"Démarrage : {len(indices)} image(s) avec « {s.model} »"
        f"{'' if Image else ' (Pillow absent : images envoyées brutes)'}.")
    captioner.start(jobs, indices, s, force)


def start_selected() -> None:
    sel = selected_indices()
    if not sel:
        ui.notify("Sélectionne une ou plusieurs images.", type="warning")
        return
    start(sel)


def start_current() -> None:
    idx = current_index()
    if idx is None:
        ui.notify("Sélectionne une image dans la liste.", type="warning")
        return
    start([idx], force=True)


def stop() -> None:
    if captioner.is_running():
        captioner.stop()
        log("Arrêt demandé, fin de l'image en cours…")


def poll_events() -> None:
    changed = False
    try:
        while True:
            ev = captioner.events.get_nowait()
            if ev[0] == "row":
                idx = ev[1]
                changed = True
                j = jobs[idx]
                if j.status == "erreur":
                    log(f"✖ {j.path.name} : {j.error}")
                elif j.status == "ok" and idx == current_index():
                    W["caption"].value = j.caption
                    W["capfile"].text = j.path.with_suffix(settings.extension).name + " (existe)"
            elif ev[0] == "phase":
                if ev[2] == "chargement":
                    log(f"Chargement du modèle « {settings.model} »…")
            elif ev[0] == "done":
                ok = sum(j.status == "ok" for j in jobs)
                err = sum(j.status == "erreur" for j in jobs)
                skip = sum(j.status == "ignoré" for j in jobs)
                log(f"{'Arrêté' if captioner.stop_event.is_set() else 'Terminé'} : "
                    f"{ok} ok, {skip} ignoré(s), {err} erreur(s).")
                W["btn_all"].enable()
                W["btn_sel"].enable()
                W["btn_stop"].disable()
                ui.notify("Captioning terminé", type="positive")
    except queue.Empty:
        pass
    snap = captioner.progress.snapshot()
    if captioner.is_running() or changed:
        W["progress"].value = snap["fraction"]
        W["progress_label"].text = f"{snap['fraction'] * 100:.0f}%"
        W["status"].text = snap["text"]
    if changed or (captioner.is_running() and int(time.time() * 5) % 5 == 0):
        refresh_table()


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
PASTE_JS = """
<script>
document.addEventListener('paste', (ev) => {
  const t = ev.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  const items = (ev.clipboardData || {}).items || [];
  for (const it of items) {
    if (it.type && it.type.startsWith('image/')) {
      const file = it.getAsFile();
      const reader = new FileReader();
      reader.onload = () => emitEvent('captionz_paste', {data: reader.result, name: file.name || 'paste.png'});
      reader.readAsDataURL(file);
      ev.preventDefault();
      return;
    }
  }
});
</script>
"""


def build() -> None:
    s = settings
    ui.page_title(APP_TITLE)
    ui.add_body_html(PASTE_JS)
    ui.on("captionz_paste", on_paste)
    W["dark"] = ui.dark_mode(value=s.dark)

    with ui.header().classes("items-center justify-between px-4 py-2"):
        ui.label("Captionz").classes("text-xl font-bold")
        with ui.row().classes("items-center gap-4"):
            W["conn"] = ui.label("…").classes("text-sm opacity-80")
            ui.switch("Mode sombre").bind_value(W["dark"], "value")

    with ui.row().classes("w-full no-wrap gap-4 p-4 items-start"):
        # ================= left column =================
        with ui.column().classes("w-3/5 gap-3"):
            with ui.card().classes("w-full"):
                ui.label("Ollama").classes("text-lg font-semibold")
                with ui.row().classes("w-full items-end gap-2"):
                    W["url"] = ui.input("URL", value=s.ollama_url).classes("w-56")
                    W["model"] = ui.select([s.model] if s.model else [], value=s.model or None,
                                           label="Modèle vision").classes("flex-grow")
                    ui.button(icon="refresh", on_click=refresh_models).props("flat round")

            with ui.card().classes("w-full"):
                ui.label("Sources").classes("text-lg font-semibold")
                with ui.row().classes("w-full items-end gap-2"):
                    W["path"] = ui.input("Chemin d'un fichier ou d'un dossier (sur cette machine)") \
                        .classes("flex-grow").on("keydown.enter", add_path_from_input)
                    W["recursive"] = ui.checkbox("récursif", value=s.recursive)
                    ui.button("Ajouter", icon="add", on_click=add_path_from_input)
                with ui.row().classes("w-full items-center gap-2"):
                    W["upload"] = ui.upload(label="Envoyer des images (copiées dans pasted/uploads)", multiple=True,
                                            auto_upload=True, on_upload=on_upload) \
                        .props('accept="image/*"').classes("flex-grow")
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("📋 Ctrl+V n'importe où dans la page pour coller une capture d'écran ou une image copiée.") \
                        .classes("text-sm opacity-70")
                    ui.space()
                    ui.button("Retirer sélection", icon="remove", on_click=remove_selected).props("flat")
                    ui.button("Vider", icon="delete", on_click=clear_jobs).props("flat")
                    W["count"] = ui.label("0 image").classes("text-sm")

            with ui.card().classes("w-full"):
                ui.label("Prompt").classes("text-lg font-semibold")
                with ui.row().classes("w-full gap-2"):
                    W["type"] = ui.select(list(CAPTION_TYPES), value=s.caption_type, label="Type de caption",
                                          on_change=update_prompt_preview).classes("flex-grow")
                    W["length"] = ui.select(list(CAPTION_LENGTHS), value=s.caption_length, label="Longueur",
                                            on_change=update_prompt_preview).classes("w-44")
                with ui.expansion("Options supplémentaires", icon="tune").classes("w-full"):
                    W["opts"] = []
                    for opt in EXTRA_OPTIONS:
                        cb = ui.checkbox(opt, value=opt in s.options, on_change=update_prompt_preview).classes("text-sm")
                        W["opts"].append((opt, cb))
                with ui.row().classes("w-full items-end gap-2"):
                    W["name"] = ui.input("Nom du personnage ({name})", value=s.name,
                                         on_change=update_prompt_preview).classes("w-72")
                    ui.label("vide = « the main character »").classes("text-sm opacity-70")
                W["custom"] = ui.textarea("Prompt personnalisé (remplace type / longueur / options si rempli)",
                                          value=s.custom_prompt, on_change=update_prompt_preview).classes("w-full").props("rows=2")
                W["preview"] = ui.textarea("Prompt final envoyé au modèle").classes("w-full").props("readonly rows=3 outlined")

            with ui.card().classes("w-full"):
                ui.label("Sortie").classes("text-lg font-semibold")
                with ui.row().classes("w-full items-end gap-2"):
                    W["prefix"] = ui.input("Préfixe (trigger)", value=s.prefix).classes("w-48")
                    W["suffix"] = ui.input("Suffixe", value=s.suffix).classes("w-48")
                    W["ext"] = ui.input("Extension", value=s.extension).classes("w-24")
                    W["single"] = ui.checkbox("Une seule ligne", value=s.single_line)
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("Captions existantes :")
                    W["existing"] = ui.radio({"skip": "ignorer", "overwrite": "écraser", "append": "ajouter à la suite"},
                                             value=s.existing).props("inline")

            with ui.card().classes("w-full"):
                ui.label("Modèle").classes("text-lg font-semibold")
                with ui.row().classes("w-full items-end gap-2"):
                    W["temp"] = ui.number("Température", value=s.temperature, min=0, max=1.5, step=0.1).classes("w-32")
                    W["keep"] = ui.input("keep_alive (0 = décharger)", value=str(s.keep_alive)).classes("w-44")
                    W["maxside"] = ui.number("Côté max px (0 = brut)", value=s.max_side, min=0, max=4096, step=128).classes("w-44")
                    W["cpu"] = ui.checkbox("Forcer CPU", value=s.cpu_only)
                    W["maxtok"] = ui.number("Tokens max (0 = illimité)", value=s.max_tokens, min=0, max=8192, step=256) \
                        .classes("w-44").tooltip("Borne la génération : un modèle qui divague est coupé au lieu de bloquer")
                    if Image is None:
                        W["maxside"].disable()
                        ui.label("(pip install pillow)").classes("text-sm opacity-70")

            with ui.card().classes("w-full"):
                with ui.row().classes("w-full items-center gap-2"):
                    W["btn_all"] = ui.button("Captionner tout", icon="play_arrow", on_click=lambda: start(None))
                    W["btn_sel"] = ui.button("Captionner la sélection", icon="playlist_play", on_click=start_selected)
                    W["btn_stop"] = ui.button("Arrêter", icon="stop", color="negative", on_click=stop)
                    W["btn_stop"].disable()
                    W["progress"] = ui.linear_progress(value=0, show_value=False).classes("flex-grow")
                    W["progress_label"] = ui.label("")
                W["status"] = ui.label("").classes("text-sm opacity-80 w-full")
                W["log"] = ui.log(max_lines=300).classes("w-full h-40 text-xs")

        # ================= right column =================
        with ui.column().classes("w-2/5 gap-3"):
            with ui.card().classes("w-full"):
                ui.label("Images").classes("text-lg font-semibold")
                columns = [
                    {"name": "name", "label": "Fichier", "field": "name", "align": "left", "sortable": True},
                    {"name": "status", "label": "Statut", "field": "status", "align": "center"},
                    {"name": "time", "label": "Durée", "field": "time", "align": "center"},
                ]
                W["table"] = ui.table(columns=columns, rows=[], row_key="id", selection="multiple",
                                      pagination=0, on_select=lambda e: show_selected()).classes("w-full").props("dense")
                W["table"].add_slot("body-cell-status", """
                    <q-td :props="props">
                      <span :class="{'text-green-7': props.value==='ok', 'text-red-7': props.value==='erreur',
                                     'text-amber-8': props.value==='ignoré', 'text-blue-7': props.value==='en cours'}">
                        {{ props.value }}
                      </span>
                    </q-td>""")
                W["table"].style("max-height: 320px")

            with ui.card().classes("w-full"):
                ui.label("Aperçu").classes("text-lg font-semibold")
                W["image"] = ui.image("").classes("w-full").style("max-height: 420px; object-fit: contain")

            with ui.card().classes("w-full"):
                ui.label("Caption (éditable)").classes("text-lg font-semibold")
                W["caption"] = ui.textarea("").classes("w-full").props("rows=7 outlined")
                with ui.row().classes("w-full items-center gap-2"):
                    ui.button("Captionner cette image", icon="play_arrow", on_click=start_current)
                    ui.button("Enregistrer la caption", icon="save", on_click=save_caption)
                    W["capfile"] = ui.label("").classes("text-sm opacity-70")

    update_prompt_preview()
    ui.timer(0.2, poll_events)
    ui.timer(0.1, refresh_models, once=True)


def main(host: str = "127.0.0.1", port: int = 8080, show: bool = True) -> None:
    ui.run(root=build, title=APP_TITLE, host=host, port=port, show=show, reload=False,
           favicon="🖼️", dark=settings.dark)


if __name__ == "__main__":
    main()
