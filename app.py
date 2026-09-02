"""
Captionz — batch image captioning through Ollama vision models.

Entry point. Two interfaces share the same core (captionz_core.py):

    python app.py               # Tkinter desktop UI (default, stdlib only)
    python app.py --ui web      # NiceGUI web UI (pip install nicegui), opens the browser
    python app.py --ui web --port 8090 --no-browser

The Ollama layer reuses patterns from crispz-studio (cz_ollama.py): vision
detection through /api/show with a name-based fallback, images downscaled to
JPEG before upload (when Pillow is installed), stripping of <think> blocks,
configurable keep_alive / CPU mode so the model does not hog VRAM.

Features:
  - Connect to an Ollama server (configurable URL), models filtered on "vision"
  - Sources: a single file, a selection of files, a folder (recursive or not),
    or an image pasted from the clipboard (Ctrl+V)
  - Composed prompt: caption type × length × checkable options × character
    name ({name}), or a custom prompt that overrides everything
  - Final prompt preview, image preview, editable caption + save
  - Existing captions: skip / overwrite / append
  - Caption everything, only the selection, or the selected image
  - Prefix/suffix (trigger word), output extension, dark mode
  - Background processing with progress, log and clean stop

Dependencies: Python 3.10+ (stdlib). Pillow optional (preview + downscaling).
The UI language is French.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from captionz_core import (  # noqa: F401  (re-exported for bench.py and older imports)
    APP_TITLE, CAPTION_LENGTHS, CAPTION_TYPES, DEFAULT_OLLAMA_URL, DEFAULT_PROMPT, EXTRA_OPTIONS,
    IMAGE_EXTS, Captioner, Job, OllamaClient, Settings, build_prompt, collect_images, save_pasted_image,
)

try:
    from PIL import Image, ImageGrab, ImageTk  # optional: preview, downscaling, paste
except ImportError:  # pragma: no cover
    Image = ImageGrab = ImageTk = None


# --------------------------------------------------------------------------- #
# Themes
# --------------------------------------------------------------------------- #
THEMES = {
    "light": dict(bg="#f3f3f3", fg="#1b1b1b", field="#ffffff", sel="#cfe3ff", border="#c8c8c8",
                  ok="#1a7f37", err="#c62828", skip="#8a6d00", run="#0b57d0", muted="#666666"),
    "dark": dict(bg="#1e1f22", fg="#e6e6e6", field="#2b2d31", sel="#3b4b66", border="#3c3f44",
                 ok="#5ed08a", err="#ff6b6b", skip="#e0c060", run="#7fb3ff", muted="#9a9a9a"),
}


class ScrollFrame(ttk.Frame):
    """Vertically scrollable frame (for the long options list)."""

    def __init__(self, parent, height=220, **kw):
        super().__init__(parent, **kw)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0, bd=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
            w.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        self.canvas.yview_scroll(int(-e.delta / 120), "units")


# --------------------------------------------------------------------------- #
# Tkinter UI
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1480x920")
        self.minsize(1100, 700)

        self.settings = Settings.load()
        self.jobs: list[Job] = []
        self.captioner = Captioner()
        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self._preview_img = None
        self._preview_path: Path | None = None
        self._text_widgets: list[tk.Text] = []
        self._last_total = 0

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self._build_ui()
        self.apply_theme(self.settings.dark)
        self._poll_ui_queue()
        self.after(200, self.refresh_models)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- construction --------------------------------------------------- #
    def _build_ui(self):
        pad = {"padx": 6, "pady": 3}
        s = self.settings
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        # ================= left column: settings =================
        top = ttk.LabelFrame(left, text="Ollama")
        top.pack(fill="x", **pad)
        ttk.Label(top, text="URL :").grid(row=0, column=0, sticky="w", **pad)
        self.var_url = tk.StringVar(value=s.ollama_url)
        ttk.Entry(top, textvariable=self.var_url, width=28).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(top, text="Modèle vision :").grid(row=0, column=2, sticky="w", **pad)
        self.var_model = tk.StringVar(value=s.model)
        self.cmb_model = ttk.Combobox(top, textvariable=self.var_model, state="readonly", width=44)
        self.cmb_model.grid(row=0, column=3, sticky="we", **pad)
        ttk.Button(top, text="↻", width=3, command=self.refresh_models).grid(row=0, column=4, **pad)
        self.lbl_conn = ttk.Label(top, text="…")
        self.lbl_conn.grid(row=0, column=5, sticky="w", **pad)
        top.columnconfigure(3, weight=1)

        # --- sources ---
        src = ttk.LabelFrame(left, text="Sources")
        src.pack(fill="x", **pad)
        ttk.Button(src, text="📄 Un fichier…", command=self.add_file).pack(side="left", **pad)
        ttk.Button(src, text="📑 Plusieurs fichiers…", command=self.add_files).pack(side="left", **pad)
        ttk.Button(src, text="📁 Un dossier…", command=self.add_folder).pack(side="left", **pad)
        ttk.Button(src, text="📋 Coller (Ctrl+V)", command=self.paste_image).pack(side="left", **pad)
        self.bind_all("<Control-v>", self._on_ctrl_v)
        self.var_recursive = tk.BooleanVar(value=s.recursive)
        ttk.Checkbutton(src, text="récursif", variable=self.var_recursive).pack(side="left", **pad)
        ttk.Separator(src, orient="vertical").pack(side="left", fill="y", padx=8, pady=4)
        ttk.Button(src, text="Retirer sélection", command=self.remove_selected).pack(side="left", **pad)
        ttk.Button(src, text="Vider", command=self.clear_jobs).pack(side="left", **pad)
        self.lbl_count = ttk.Label(src, text="0 image")
        self.lbl_count.pack(side="right", **pad)

        # --- composed prompt ---
        pf = ttk.LabelFrame(left, text="Prompt")
        pf.pack(fill="both", expand=True, **pad)
        row = ttk.Frame(pf)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Type de caption :").pack(side="left")
        self.var_type = tk.StringVar(value=s.caption_type)
        cb = ttk.Combobox(row, textvariable=self.var_type, state="readonly", width=30, values=list(CAPTION_TYPES))
        cb.pack(side="left", padx=4)
        ttk.Label(row, text="Longueur :").pack(side="left", padx=(12, 0))
        self.var_length = tk.StringVar(value=s.caption_length)
        cl = ttk.Combobox(row, textvariable=self.var_length, state="readonly", width=14, values=list(CAPTION_LENGTHS))
        cl.pack(side="left", padx=4)
        for w in (cb, cl):
            w.bind("<<ComboboxSelected>>", self._update_prompt_preview)

        ttk.Label(pf, text="Options supplémentaires :").pack(anchor="w", padx=6)
        sf = ScrollFrame(pf, height=190)
        sf.pack(fill="x", padx=6)
        self.opt_vars: list[tuple[str, tk.BooleanVar]] = []
        for opt in EXTRA_OPTIONS:
            v = tk.BooleanVar(value=opt in s.options)
            v.trace_add("write", lambda *_: self._update_prompt_preview())
            ttk.Checkbutton(sf.inner, text=opt, variable=v).pack(anchor="w", padx=2)
            self.opt_vars.append((opt, v))

        row = ttk.Frame(pf)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Nom du personnage ({name}) :").pack(side="left")
        self.var_name = tk.StringVar(value=s.name)
        self.var_name.trace_add("write", lambda *_: self._update_prompt_preview())
        ttk.Entry(row, textvariable=self.var_name, width=30).pack(side="left", padx=4)
        ttk.Label(row, text="vide = « the main character »").pack(side="left", padx=4)

        ttk.Label(pf, text="Prompt personnalisé (remplace type / longueur / options si rempli) :").pack(anchor="w", padx=6)
        self.txt_custom = tk.Text(pf, height=3, wrap="word")
        self.txt_custom.pack(fill="x", padx=6)
        self.txt_custom.insert("1.0", s.custom_prompt)
        self.txt_custom.bind("<KeyRelease>", self._update_prompt_preview)
        self._text_widgets.append(self.txt_custom)

        ttk.Label(pf, text="Prompt final envoyé au modèle :").pack(anchor="w", padx=6, pady=(4, 0))
        self.txt_preview = tk.Text(pf, height=4, wrap="word", state="disabled")
        self.txt_preview.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        self._text_widgets.append(self.txt_preview)

        # --- output ---
        of = ttk.LabelFrame(left, text="Sortie")
        of.pack(fill="x", **pad)
        self.var_prefix = tk.StringVar(value=s.prefix)
        self.var_suffix = tk.StringVar(value=s.suffix)
        self.var_ext = tk.StringVar(value=s.extension)
        self.var_existing = tk.StringVar(value=s.existing)
        self.var_single = tk.BooleanVar(value=s.single_line)
        ttk.Label(of, text="Préfixe (trigger) :").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(of, textvariable=self.var_prefix, width=24).grid(row=0, column=1, **pad)
        ttk.Label(of, text="Suffixe :").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(of, textvariable=self.var_suffix, width=24).grid(row=0, column=3, **pad)
        ttk.Label(of, text="Extension :").grid(row=0, column=4, sticky="w", **pad)
        ttk.Entry(of, textvariable=self.var_ext, width=8).grid(row=0, column=5, **pad)
        ttk.Label(of, text="Captions existantes :").grid(row=1, column=0, sticky="w", **pad)
        rf = ttk.Frame(of)
        rf.grid(row=1, column=1, columnspan=3, sticky="w")
        for txt, val in (("ignorer", "skip"), ("écraser", "overwrite"), ("ajouter à la suite", "append")):
            ttk.Radiobutton(rf, text=txt, value=val, variable=self.var_existing).pack(side="left", padx=4)
        ttk.Checkbutton(of, text="Une seule ligne", variable=self.var_single).grid(row=1, column=4, columnspan=2, sticky="w", **pad)

        # --- model / perf ---
        mf = ttk.LabelFrame(left, text="Modèle")
        mf.pack(fill="x", **pad)
        self.var_temp = tk.DoubleVar(value=s.temperature)
        self.var_maxtok = tk.IntVar(value=s.max_tokens)
        self.var_keep = tk.StringVar(value=str(s.keep_alive))
        self.var_maxside = tk.IntVar(value=s.max_side)
        self.var_cpu = tk.BooleanVar(value=s.cpu_only)
        ttk.Label(mf, text="Température :").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(mf, from_=0.0, to=1.5, increment=0.1, textvariable=self.var_temp, width=6).grid(row=0, column=1, **pad)
        ttk.Label(mf, text="keep_alive (0 = décharger) :").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(mf, textvariable=self.var_keep, width=8).grid(row=0, column=3, **pad)
        ttk.Label(mf, text="Côté max px (0 = brut) :").grid(row=0, column=4, sticky="w", **pad)
        sb = ttk.Spinbox(mf, from_=0, to=4096, increment=128, textvariable=self.var_maxside, width=7)
        sb.grid(row=0, column=5, **pad)
        if Image is None:
            sb.configure(state="disabled")
            ttk.Label(mf, text="(pip install pillow)").grid(row=0, column=6, sticky="w")
        ttk.Checkbutton(mf, text="Forcer CPU", variable=self.var_cpu).grid(row=0, column=7, sticky="w", **pad)
        ttk.Label(mf, text="Tokens max (0 = illimité) :").grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(mf, from_=0, to=8192, increment=256, textvariable=self.var_maxtok, width=6).grid(row=1, column=1, **pad)
        ttk.Label(mf, text="borne la génération : un modèle qui divague est coupé au lieu de bloquer").grid(
            row=1, column=2, columnspan=6, sticky="w", **pad)

        # --- controls ---
        ctl = ttk.Frame(left)
        ctl.pack(fill="x", **pad)
        self.btn_start = ttk.Button(ctl, text="▶ Captionner tout", command=lambda: self.start(None))
        self.btn_start.pack(side="left", **pad)
        self.btn_sel = ttk.Button(ctl, text="▶ Captionner la sélection", command=self.start_selected)
        self.btn_sel.pack(side="left", **pad)
        self.btn_stop = ttk.Button(ctl, text="■ Arrêter", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", **pad)
        ttk.Button(ctl, text="🌓 Mode sombre", command=self.toggle_theme).pack(side="right", **pad)
        self.progress = ttk.Progressbar(ctl, mode="determinate", maximum=1000)
        self.progress.pack(side="left", fill="x", expand=True, **pad)
        self.lbl_progress = ttk.Label(ctl, text="")
        self.lbl_progress.pack(side="left", **pad)
        self.lbl_status = ttk.Label(left, text="", anchor="w")
        self.lbl_status.pack(fill="x", padx=12, pady=(0, 2))

        self.log = scrolledtext.ScrolledText(left, height=5, state="disabled", wrap="word")
        self.log.pack(fill="x", padx=6, pady=(0, 6))
        self._text_widgets.append(self.log)

        # ================= right column: images =================
        lf = ttk.LabelFrame(right, text="Images")
        lf.pack(fill="both", expand=True, **pad)
        cols = ("file", "status", "time")
        self.tree = ttk.Treeview(lf, columns=cols, show="headings", selectmode="extended", height=10)
        self.tree.heading("file", text="Fichier")
        self.tree.heading("status", text="Statut")
        self.tree.heading("time", text="Durée")
        self.tree.column("file", width=340, anchor="w")
        self.tree.column("status", width=80, anchor="center")
        self.tree.column("time", width=60, anchor="center")
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        pv = ttk.LabelFrame(right, text="Aperçu")
        pv.pack(fill="both", expand=True, **pad)
        self.canvas = tk.Canvas(pv, height=300, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._show_preview())

        cf = ttk.LabelFrame(right, text="Caption (éditable)")
        cf.pack(fill="both", expand=True, **pad)
        self.txt_caption = tk.Text(cf, height=7, wrap="word")
        self.txt_caption.pack(fill="both", expand=True, padx=6, pady=4)
        self._text_widgets.append(self.txt_caption)
        bf = ttk.Frame(cf)
        bf.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Button(bf, text="▶ Captionner cette image", command=self.start_current).pack(side="left")
        ttk.Button(bf, text="💾 Enregistrer la caption", command=self.save_caption).pack(side="left", padx=6)
        self.lbl_caption_file = ttk.Label(bf, text="")
        self.lbl_caption_file.pack(side="left", padx=6)

        self._update_prompt_preview()

    # ---- theme ------------------------------------------------------------ #
    def apply_theme(self, dark: bool):
        t = THEMES["dark" if dark else "light"]
        self.settings.dark = dark
        self.configure(bg=t["bg"])
        st = self.style
        st.configure(".", background=t["bg"], foreground=t["fg"], fieldbackground=t["field"],
                     bordercolor=t["border"], lightcolor=t["bg"], darkcolor=t["bg"], troughcolor=t["field"])
        for w in ("TFrame", "TLabel", "TLabelframe", "TCheckbutton", "TRadiobutton", "TPanedwindow"):
            st.configure(w, background=t["bg"], foreground=t["fg"])
        st.configure("TLabelframe.Label", background=t["bg"], foreground=t["fg"])
        st.configure("TButton", background=t["field"], foreground=t["fg"])
        st.map("TButton", background=[("active", t["sel"])])
        st.map("TCheckbutton", background=[("active", t["bg"])])
        st.map("TRadiobutton", background=[("active", t["bg"])])
        st.configure("TEntry", fieldbackground=t["field"], foreground=t["fg"], insertcolor=t["fg"])
        st.configure("TCombobox", fieldbackground=t["field"], foreground=t["fg"], background=t["field"],
                     arrowcolor=t["fg"])
        st.map("TCombobox", fieldbackground=[("readonly", t["field"])], foreground=[("readonly", t["fg"])])
        st.configure("TSpinbox", fieldbackground=t["field"], foreground=t["fg"], arrowcolor=t["fg"])
        st.configure("Treeview", background=t["field"], fieldbackground=t["field"], foreground=t["fg"])
        st.configure("Treeview.Heading", background=t["bg"], foreground=t["fg"])
        st.map("Treeview", background=[("selected", t["sel"])], foreground=[("selected", t["fg"])])
        st.configure("TScrollbar", background=t["field"], troughcolor=t["bg"], arrowcolor=t["fg"])
        st.configure("Horizontal.TProgressbar", background=t["run"], troughcolor=t["field"])
        self.option_add("*TCombobox*Listbox.background", t["field"])
        self.option_add("*TCombobox*Listbox.foreground", t["fg"])
        for w in self._text_widgets:
            w.configure(bg=t["field"], fg=t["fg"], insertbackground=t["fg"], selectbackground=t["sel"],
                        highlightthickness=1, highlightbackground=t["border"], relief="flat")
        self.canvas.configure(bg=t["field"])
        for c in self._all_children(self, tk.Canvas):
            if c is not self.canvas:
                c.configure(bg=t["bg"])
        self.tree.tag_configure("ok", foreground=t["ok"])
        self.tree.tag_configure("err", foreground=t["err"])
        self.tree.tag_configure("skip", foreground=t["skip"])
        self.tree.tag_configure("run", foreground=t["run"])
        self._theme = t
        self._show_preview()

    def toggle_theme(self):
        self.apply_theme(not self.settings.dark)

    @staticmethod
    def _all_children(widget, cls):
        out = []
        for c in widget.winfo_children():
            if isinstance(c, cls):
                out.append(c)
            out.extend(App._all_children(c, cls))
        return out

    # ---- UI helpers ------------------------------------------------------ #
    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", time.strftime("[%H:%M:%S] ") + msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _update_prompt_preview(self, _event=None):
        try:
            p = self._collect_settings().prompt
        except Exception:
            return
        self.txt_preview.configure(state="normal")
        self.txt_preview.delete("1.0", "end")
        self.txt_preview.insert("1.0", p)
        self.txt_preview.configure(state="disabled")

    def _collect_settings(self) -> Settings:
        return Settings(
            ollama_url=self.var_url.get().strip() or DEFAULT_OLLAMA_URL,
            model=self.var_model.get().strip(),
            caption_type=self.var_type.get(),
            caption_length=self.var_length.get(),
            options=[o for o, v in self.opt_vars if v.get()],
            name=self.var_name.get(),
            custom_prompt=self.txt_custom.get("1.0", "end").strip(),
            prefix=self.var_prefix.get(),
            suffix=self.var_suffix.get(),
            extension=self.var_ext.get().strip() or ".txt",
            recursive=self.var_recursive.get(),
            existing=self.var_existing.get(),
            temperature=float(self.var_temp.get()),
            max_tokens=int(self.var_maxtok.get() or 0),
            single_line=self.var_single.get(),
            keep_alive=self.var_keep.get().strip() or "0",
            max_side=int(self.var_maxside.get() or 0),
            cpu_only=self.var_cpu.get(),
            dark=self.settings.dark,
            paste_dir=self.settings.paste_dir,
            vision_blocklist=list(self.settings.vision_blocklist),
        ).normalized()

    def _update_count(self):
        n = len(self.jobs)
        self.lbl_count.configure(text=f"{n} image{'s' if n > 1 else ''}")

    def _refresh_row(self, idx: int):
        job = self.jobs[idx]
        tag = {"ok": "ok", "erreur": "err", "ignoré": "skip", "en cours": "run"}.get(job.status, "")
        if job.status == "en cours" and job.started:
            dur = f"{time.time() - job.started:.0f}s…"
        else:
            dur = f"{job.duration:.1f}s" if job.duration else ""
        self.tree.item(str(idx), values=(str(job.path), job.status, dur), tags=(tag,))

    def _selected_indices(self) -> list[int]:
        return [int(i) for i in self.tree.selection()]

    def _current_index(self) -> int | None:
        sel = self._selected_indices()
        return sel[0] if sel else None

    # ---- preview + editable caption -------------------------------------- #
    def _on_select(self, _event=None):
        idx = self._current_index()
        if idx is None:
            return
        job = self.jobs[idx]
        self._show_preview(job.path)
        text = job.caption
        cap_file = job.path.with_suffix(self._collect_settings().extension)
        if not text and cap_file.exists():
            try:
                text = cap_file.read_text("utf-8").strip()
            except Exception:
                text = ""
        self.txt_caption.delete("1.0", "end")
        self.txt_caption.insert("1.0", text)
        self.lbl_caption_file.configure(text=cap_file.name + (" (existe)" if cap_file.exists() else ""))

    def _show_preview(self, path: Path | None = None):
        if path is not None:
            self._preview_path = path
        path = self._preview_path
        self.canvas.delete("all")
        w, h = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        t = getattr(self, "_theme", THEMES["light"])
        if path is None:
            self.canvas.create_text(w / 2, h / 2, text="Aucune image sélectionnée", fill=t["muted"])
            return
        try:
            if Image is not None:
                img = Image.open(path)
                img.thumbnail((w - 8, h - 8))
                self._preview_img = ImageTk.PhotoImage(img)
            else:
                img = tk.PhotoImage(file=str(path))  # PNG/GIF only
                f = max(1, int(max(img.width() / (w - 8), img.height() / (h - 8))) + 1)
                self._preview_img = img.subsample(f, f)
            self.canvas.create_image(w / 2, h / 2, image=self._preview_img)
        except Exception as e:  # noqa: BLE001
            self.canvas.create_text(w / 2, h / 2, text=f"Aperçu impossible\n{e}", fill=t["muted"], justify="center")

    def save_caption(self):
        idx = self._current_index()
        if idx is None:
            messagebox.showinfo(APP_TITLE, "Sélectionne une image dans la liste.")
            return
        job = self.jobs[idx]
        text = self.txt_caption.get("1.0", "end").strip()
        out = job.path.with_suffix(self._collect_settings().extension)
        out.write_text(text + "\n", encoding="utf-8")
        job.caption = text
        if job.status != "erreur":
            job.status = "ok"
        self._refresh_row(idx)
        self.lbl_caption_file.configure(text=out.name + " (existe)")
        self._log(f"💾 {out.name} enregistré.")

    # ---- models ---------------------------------------------------------- #
    def refresh_models(self):
        url = self.var_url.get().strip() or DEFAULT_OLLAMA_URL
        self.lbl_conn.configure(text="connexion…")

        def work():
            try:
                models = OllamaClient(url, timeout=15).list_vision_models(self.settings.vision_blocklist)
                self.ui_queue.put(("models", models, None))
            except Exception as e:  # noqa: BLE001
                self.ui_queue.put(("models", [], str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _apply_models(self, models: list[str], error: str | None):
        if error:
            self.lbl_conn.configure(text="✖ hors ligne")
            self._log(f"Impossible de joindre Ollama : {error}")
            self.cmb_model["values"] = []
            return
        self.cmb_model["values"] = models
        if models:
            self.lbl_conn.configure(text=f"✔ {len(models)} modèle(s) vision")
            if self.var_model.get() not in models:
                self.var_model.set(models[0])
        else:
            self.lbl_conn.configure(text="aucun modèle vision")
            self._log("Aucun modèle vision trouvé. Exemple : `ollama pull qwen3-vl:8b`.")

    # ---- sources --------------------------------------------------------- #
    def _add_paths(self, paths: list[Path]):
        images = collect_images(paths, self.var_recursive.get())
        existing = {j.path for j in self.jobs}
        added = 0
        for img in images:
            if img in existing:
                continue
            self.jobs.append(Job(img))
            idx = len(self.jobs) - 1
            self.tree.insert("", "end", iid=str(idx), values=(str(img), "en attente", ""))
            added += 1
        self._update_count()
        self._log(f"{added} image(s) ajoutée(s) ({len(images) - added} doublon(s) ignoré(s)).")
        if added and not self.tree.selection():
            first = str(len(self.jobs) - added)
            self.tree.selection_set(first)
            self.tree.see(first)

    def add_file(self):
        f = filedialog.askopenfilename(title="Choisir une image", filetypes=self._filetypes())
        if f:
            self._add_paths([Path(f)])

    def add_files(self):
        fs = filedialog.askopenfilenames(title="Choisir des images", filetypes=self._filetypes())
        if fs:
            self._add_paths([Path(f) for f in fs])

    def add_folder(self):
        d = filedialog.askdirectory(title="Choisir un dossier d'images")
        if d:
            self._add_paths([Path(d)])

    @staticmethod
    def _filetypes():
        pat = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        return [("Images", pat), ("Tous les fichiers", "*.*")]

    # ---- paste from clipboard -------------------------------------------- #
    def _on_ctrl_v(self, event):
        # keep the normal paste behaviour inside text fields
        if isinstance(event.widget, (tk.Text, tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox)):
            return
        self.paste_image()

    def paste_image(self):
        """Clipboard -> list. Three cases: a bitmap (screenshot, browser "copy
        image") saved as PNG in the paste folder; files copied in the file
        explorer; or a path pasted as text."""
        paths: list[Path] = []
        data = None
        if ImageGrab is not None:
            try:
                data = ImageGrab.grabclipboard()
            except Exception as e:  # noqa: BLE001
                self._log(f"Presse-papiers illisible : {e}")
        if isinstance(data, list):                       # files copied in the explorer
            paths = [Path(p) for p in data]
        elif data is not None and Image is not None and isinstance(data, Image.Image):
            out = save_pasted_image(data, self.settings.paste_path)
            paths = [out]
            self._log(f"Image collée enregistrée : {out}")
        else:                                            # text: file path(s)
            try:
                txt = self.clipboard_get()
            except tk.TclError:
                txt = ""
            for line in txt.replace('"', "").splitlines():
                p = Path(line.strip())
                if line.strip() and p.exists():
                    paths.append(p)
        if not paths:
            msg = "Aucune image dans le presse-papiers."
            if ImageGrab is None:
                msg += " Installe Pillow (pip install pillow) pour coller des captures d'écran."
            self._log(msg)
            return
        self._add_paths(paths)
        last = str(len(self.jobs) - 1)
        self.tree.selection_set(last)
        self.tree.see(last)
        self._on_select()

    def remove_selected(self):
        if self.captioner.is_running():
            return
        sel = set(self._selected_indices())
        if not sel:
            return
        self.jobs = [j for i, j in enumerate(self.jobs) if i not in sel]
        self._rebuild_tree()

    def clear_jobs(self):
        if self.captioner.is_running():
            return
        self.jobs.clear()
        self._rebuild_tree()
        self._preview_path = None
        self._show_preview()
        self.txt_caption.delete("1.0", "end")
        self.lbl_caption_file.configure(text="")

    def _rebuild_tree(self):
        self.tree.delete(*self.tree.get_children())
        for idx, job in enumerate(self.jobs):
            self.tree.insert("", "end", iid=str(idx), values=("", "", ""))
            self._refresh_row(idx)
        self._update_count()

    # ---- run ------------------------------------------------------------- #
    def start_selected(self):
        sel = self._selected_indices()
        if not sel:
            messagebox.showinfo(APP_TITLE, "Sélectionne une ou plusieurs images dans la liste.")
            return
        self.start(sel)

    def start_current(self):
        idx = self._current_index()
        if idx is None:
            messagebox.showinfo(APP_TITLE, "Sélectionne une image dans la liste.")
            return
        self.start([idx], force=True)  # explicitly requested: always overwrite

    def start(self, indices: list[int] | None, force: bool = False):
        if self.captioner.is_running():
            return
        s = self._collect_settings()
        if not s.model:
            messagebox.showwarning(APP_TITLE, "Sélectionne un modèle vision.")
            return
        if not self.jobs:
            messagebox.showwarning(APP_TITLE, "Ajoute au moins une image ou un dossier.")
            return
        if indices is None:
            indices = list(range(len(self.jobs)))
        self.settings = s
        s.save()
        for b in (self.btn_start, self.btn_sel):
            b.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress.configure(value=0)
        self.lbl_progress.configure(text="0%")
        self.lbl_status.configure(text="démarrage…")
        self._log(f"Démarrage : {len(indices)} image(s) avec « {s.model} »"
                  f"{'' if Image else ' (Pillow absent : images envoyées brutes)'}.")
        self.captioner.start(self.jobs, indices, s, force)

    def stop(self):
        if self.captioner.is_running():
            self.captioner.stop()
            self._log("Arrêt demandé, fin de l'image en cours…")

    # ---- UI loop --------------------------------------------------------- #
    def _poll_ui_queue(self):
        try:
            while True:
                kind, a, b = self.ui_queue.get_nowait()
                if kind == "models":
                    self._apply_models(a, b)
        except queue.Empty:
            pass
        try:
            while True:
                ev = self.captioner.events.get_nowait()
                if ev[0] == "row":
                    idx = ev[1]
                    self._refresh_row(idx)
                    job = self.jobs[idx]
                    if job.status == "erreur":
                        self._log(f"✖ {job.path.name} : {job.error}")
                    elif job.status == "ok" and idx == self._current_index():
                        self.txt_caption.delete("1.0", "end")
                        self.txt_caption.insert("1.0", job.caption)
                        self.lbl_caption_file.configure(
                            text=job.path.with_suffix(self.settings.extension).name + " (existe)")
                elif ev[0] == "phase":
                    if ev[2] == "chargement":
                        self._log(f"Chargement du modèle « {self.settings.model} »…")
                elif ev[0] == "done":
                    self._on_done()
        except queue.Empty:
            pass
        if self.captioner.is_running():  # live status: phase, timer, %, ETA
            snap = self.captioner.progress.snapshot()
            self.progress.configure(value=int(snap["fraction"] * 1000))
            self.lbl_progress.configure(text=f"{snap['fraction'] * 100:.0f}%")
            self.lbl_status.configure(text=snap["text"])
            for i, j in enumerate(self.jobs):
                if j.status == "en cours":
                    self._refresh_row(i)
        self.after(100, self._poll_ui_queue)

    def _on_done(self):
        ok = sum(j.status == "ok" for j in self.jobs)
        err = sum(j.status == "erreur" for j in self.jobs)
        skip = sum(j.status == "ignoré" for j in self.jobs)
        stopped = self.captioner.stop_event.is_set()
        snap = self.captioner.progress.snapshot()
        self.progress.configure(value=int(snap["fraction"] * 1000))
        self.lbl_progress.configure(text=f"{snap['fraction'] * 100:.0f}%")
        self.lbl_status.configure(text=snap["text"])
        self._log(f"{'Arrêté' if stopped else 'Terminé'} : {ok} ok, {skip} ignoré(s), {err} erreur(s) · {snap['text']}")
        for b in (self.btn_start, self.btn_sel):
            b.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _on_close(self):
        if self.captioner.is_running():
            if not messagebox.askyesno(APP_TITLE, "Un traitement est en cours. Quitter quand même ?"):
                return
            self.captioner.stop()
        try:
            self._collect_settings().save()
        except Exception:
            pass
        self.destroy()


# --------------------------------------------------------------------------- #
# Entry point: choose the UI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Captionz — Ollama vision captioning")
    ap.add_argument("--ui", choices=["tk", "web"], default="tk",
                    help="tk = Tkinter desktop UI (default), web = NiceGUI web UI")
    ap.add_argument("--port", type=int, default=8080, help="web UI port (default 8080)")
    ap.add_argument("--host", default="127.0.0.1", help="web UI host (default 127.0.0.1; 0.0.0.0 to expose on the LAN)")
    ap.add_argument("--no-browser", action="store_true", help="web UI: do not open the browser automatically")
    a = ap.parse_args(argv)
    if a.ui == "web":
        try:
            import webui
        except ImportError as e:
            sys.exit(f"NiceGUI is not installed ({e}). Run: pip install nicegui")
        webui.main(host=a.host, port=a.port, show=not a.no_browser)
    else:
        App().mainloop()


if __name__ == "__main__":
    main()
