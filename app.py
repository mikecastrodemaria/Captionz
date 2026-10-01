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
The UI language is English.
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
    APP_TITLE, BACKEND_LABELS, BACKENDS, CAPTION_LENGTHS, CAPTION_TYPES, DEFAULT_OLLAMA_URL, DEFAULT_PROMPT,
    EXTRA_OPTIONS, IMAGE_EXTS, Captioner, Job, OllamaClient, Settings, active_model, build_prompt, collect_images,
    make_backend, save_pasted_image,
)
from captionz_llamacpp import DEFAULT_MODEL as LC_DEFAULT, KNOWN_MODELS as LC_KNOWN, ModelRegistry, ServerBinary

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
        top = ttk.LabelFrame(left, text="Backend")
        top.pack(fill="x", **pad)
        self.var_backend_label = tk.StringVar(value=BACKEND_LABELS.get(s.backend, BACKEND_LABELS["ollama"]))
        cbb = ttk.Combobox(top, textvariable=self.var_backend_label, state="readonly", width=22,
                           values=[BACKEND_LABELS[b] for b in BACKENDS])
        cbb.grid(row=0, column=0, sticky="w", **pad)
        cbb.bind("<<ComboboxSelected>>", lambda e: self._on_backend_change())
        self.lbl_url = ttk.Label(top, text="URL:")
        self.lbl_url.grid(row=0, column=1, sticky="w", **pad)
        self.var_url = tk.StringVar(value=s.ollama_url)
        self.ent_url = ttk.Entry(top, textvariable=self.var_url, width=24)
        self.ent_url.grid(row=0, column=2, sticky="w", **pad)
        ttk.Label(top, text="Model:").grid(row=0, column=3, sticky="w", **pad)
        self.var_model = tk.StringVar(value=active_model(s))
        self.cmb_model = ttk.Combobox(top, textvariable=self.var_model, state="readonly", width=40)
        self.cmb_model.grid(row=0, column=4, sticky="we", **pad)
        ttk.Button(top, text="↻", width=3, command=self.refresh_models).grid(row=0, column=5, **pad)
        self.btn_models = ttk.Button(top, text="llama.cpp models…", command=self.open_model_manager)
        self.btn_models.grid(row=0, column=6, **pad)
        self.lbl_conn = ttk.Label(top, text="…")
        self.lbl_conn.grid(row=0, column=7, sticky="w", **pad)
        top.columnconfigure(4, weight=1)
        self._models_by_backend: dict[str, str] = {"ollama": s.model, "llamacpp": s.llamacpp_model, "hf": s.hf_model}
        self._on_backend_change(initial=True)

        # --- sources ---
        src = ttk.LabelFrame(left, text="Sources")
        src.pack(fill="x", **pad)
        ttk.Button(src, text="📄 Add file…", command=self.add_file).pack(side="left", **pad)
        ttk.Button(src, text="📑 Add files…", command=self.add_files).pack(side="left", **pad)
        ttk.Button(src, text="📁 Add folder…", command=self.add_folder).pack(side="left", **pad)
        ttk.Button(src, text="📋 Paste (Ctrl+V)", command=self.paste_image).pack(side="left", **pad)
        self.bind_all("<Control-v>", self._on_ctrl_v)
        self.var_recursive = tk.BooleanVar(value=s.recursive)
        ttk.Checkbutton(src, text="Recursive", variable=self.var_recursive).pack(side="left", **pad)
        ttk.Separator(src, orient="vertical").pack(side="left", fill="y", padx=8, pady=4)
        ttk.Button(src, text="Remove selected", command=self.remove_selected).pack(side="left", **pad)
        ttk.Button(src, text="Clear", command=self.clear_jobs).pack(side="left", **pad)
        self.lbl_count = ttk.Label(src, text="0 image")
        self.lbl_count.pack(side="right", **pad)

        # --- composed prompt ---
        pf = ttk.LabelFrame(left, text="Prompt")
        pf.pack(fill="both", expand=True, **pad)
        row = ttk.Frame(pf)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Caption type:").pack(side="left")
        self.var_type = tk.StringVar(value=s.caption_type)
        cb = ttk.Combobox(row, textvariable=self.var_type, state="readonly", width=30, values=list(CAPTION_TYPES))
        cb.pack(side="left", padx=4)
        ttk.Label(row, text="Length:").pack(side="left", padx=(12, 0))
        self.var_length = tk.StringVar(value=s.caption_length)
        cl = ttk.Combobox(row, textvariable=self.var_length, state="readonly", width=14, values=list(CAPTION_LENGTHS))
        cl.pack(side="left", padx=4)
        for w in (cb, cl):
            w.bind("<<ComboboxSelected>>", self._update_prompt_preview)

        ttk.Label(pf, text="Additional options:").pack(anchor="w", padx=6)
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
        ttk.Label(row, text="Character name ({name}):").pack(side="left")
        self.var_name = tk.StringVar(value=s.name)
        self.var_name.trace_add("write", lambda *_: self._update_prompt_preview())
        ttk.Entry(row, textvariable=self.var_name, width=30).pack(side="left", padx=4)
        ttk.Label(row, text="blank = “the main character”").pack(side="left", padx=4)

        ttk.Label(pf, text="Custom prompt (overrides type, length, and options when provided):").pack(anchor="w", padx=6)
        self.txt_custom = tk.Text(pf, height=3, wrap="word")
        self.txt_custom.pack(fill="x", padx=6)
        self.txt_custom.insert("1.0", s.custom_prompt)
        self.txt_custom.bind("<KeyRelease>", self._update_prompt_preview)
        self._text_widgets.append(self.txt_custom)

        ttk.Label(pf, text="Final prompt sent to the model:").pack(anchor="w", padx=6, pady=(4, 0))
        self.txt_preview = tk.Text(pf, height=4, wrap="word", state="disabled")
        self.txt_preview.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        self._text_widgets.append(self.txt_preview)

        # --- output ---
        of = ttk.LabelFrame(left, text="Output")
        of.pack(fill="x", **pad)
        self.var_prefix = tk.StringVar(value=s.prefix)
        self.var_suffix = tk.StringVar(value=s.suffix)
        self.var_ext = tk.StringVar(value=s.extension)
        self.var_existing = tk.StringVar(value=s.existing)
        self.var_single = tk.BooleanVar(value=s.single_line)
        ttk.Label(of, text="Prefix (trigger):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(of, textvariable=self.var_prefix, width=24).grid(row=0, column=1, **pad)
        ttk.Label(of, text="Suffix:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(of, textvariable=self.var_suffix, width=24).grid(row=0, column=3, **pad)
        ttk.Label(of, text="Extension:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Entry(of, textvariable=self.var_ext, width=8).grid(row=0, column=5, **pad)
        ttk.Label(of, text="Existing captions:").grid(row=1, column=0, sticky="w", **pad)
        rf = ttk.Frame(of)
        rf.grid(row=1, column=1, columnspan=3, sticky="w")
        for txt, val in (("Skip", "skip"), ("Overwrite", "overwrite"), ("Append", "append")):
            ttk.Radiobutton(rf, text=txt, value=val, variable=self.var_existing).pack(side="left", padx=4)
        ttk.Checkbutton(of, text="Single line", variable=self.var_single).grid(row=1, column=4, columnspan=2, sticky="w", **pad)

        # --- model / perf ---
        mf = ttk.LabelFrame(left, text="Model")
        mf.pack(fill="x", **pad)
        self.var_temp = tk.DoubleVar(value=s.temperature)
        self.var_maxtok = tk.IntVar(value=s.max_tokens)
        self.var_nothink = tk.BooleanVar(value=s.no_think)
        self.var_keep = tk.StringVar(value=str(s.keep_alive))
        self.var_maxside = tk.IntVar(value=s.max_side)
        self.var_cpu = tk.BooleanVar(value=s.cpu_only)
        ttk.Label(mf, text="Temperature:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(mf, from_=0.0, to=1.5, increment=0.1, textvariable=self.var_temp, width=6).grid(row=0, column=1, **pad)
        ttk.Label(mf, text="keep_alive (0 = unload):").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(mf, textvariable=self.var_keep, width=8).grid(row=0, column=3, **pad)
        ttk.Label(mf, text="Maximum side in px (0 = original):").grid(row=0, column=4, sticky="w", **pad)
        sb = ttk.Spinbox(mf, from_=0, to=4096, increment=128, textvariable=self.var_maxside, width=7)
        sb.grid(row=0, column=5, **pad)
        if Image is None:
            sb.configure(state="disabled")
            ttk.Label(mf, text="(pip install pillow)").grid(row=0, column=6, sticky="w")
        ttk.Checkbutton(mf, text="Force CPU", variable=self.var_cpu).grid(row=0, column=7, sticky="w", **pad)
        ttk.Label(mf, text="Maximum tokens (0 = unlimited):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Spinbox(mf, from_=0, to=8192, increment=256, textvariable=self.var_maxtok, width=6).grid(row=1, column=1, **pad)
        ttk.Label(mf, text="Limits generation to prevent rambling models from hanging").grid(
            row=1, column=2, columnspan=5, sticky="w", **pad)
        ttk.Checkbutton(mf, text="Disable reasoning", variable=self.var_nothink).grid(row=1, column=7, sticky="w", **pad)

        # --- controls ---
        ctl = ttk.Frame(left)
        ctl.pack(fill="x", **pad)
        self.btn_start = ttk.Button(ctl, text="▶ Caption all", command=lambda: self.start(None))
        self.btn_start.pack(side="left", **pad)
        self.btn_sel = ttk.Button(ctl, text="▶ Caption selected", command=self.start_selected)
        self.btn_sel.pack(side="left", **pad)
        self.btn_stop = ttk.Button(ctl, text="■ Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", **pad)
        ttk.Button(ctl, text="🌓 Dark mode", command=self.toggle_theme).pack(side="right", **pad)
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
        self.tree.heading("file", text="File")
        self.tree.heading("status", text="Status")
        self.tree.heading("time", text="Duration")
        self.tree.column("file", width=340, anchor="w")
        self.tree.column("status", width=80, anchor="center")
        self.tree.column("time", width=60, anchor="center")
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        pv = ttk.LabelFrame(right, text="Preview")
        pv.pack(fill="both", expand=True, **pad)
        self.canvas = tk.Canvas(pv, height=300, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._show_preview())

        cf = ttk.LabelFrame(right, text="Caption (editable)")
        cf.pack(fill="both", expand=True, **pad)
        self.txt_caption = tk.Text(cf, height=7, wrap="word")
        self.txt_caption.pack(fill="both", expand=True, padx=6, pady=4)
        self._text_widgets.append(self.txt_caption)
        bf = ttk.Frame(cf)
        bf.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Button(bf, text="▶ Caption this image", command=self.start_current).pack(side="left")
        ttk.Button(bf, text="💾 Save caption", command=self.save_caption).pack(side="left", padx=6)
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
            backend=self.backend,
            ollama_url=self.var_url.get().strip() or DEFAULT_OLLAMA_URL,
            model=self._models_by_backend.get("ollama", ""),
            llamacpp_model=self._models_by_backend.get("llamacpp", ""),
            hf_model=self._models_by_backend.get("hf", ""),
            llamacpp_dir=self.settings.llamacpp_dir,
            llamacpp_build=self.settings.llamacpp_build,
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
            no_think=self.var_nothink.get(),
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
        tag = {"ok": "ok", "error": "err", "skipped": "skip", "processing": "run"}.get(job.status, "")
        if job.status == "processing" and job.started:
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
        self.lbl_caption_file.configure(text=cap_file.name + (" (exists)" if cap_file.exists() else ""))

    def _show_preview(self, path: Path | None = None):
        if path is not None:
            self._preview_path = path
        path = self._preview_path
        self.canvas.delete("all")
        w, h = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        t = getattr(self, "_theme", THEMES["light"])
        if path is None:
            self.canvas.create_text(w / 2, h / 2, text="No image selected", fill=t["muted"])
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
            self.canvas.create_text(w / 2, h / 2, text=f"Preview unavailable\n{e}", fill=t["muted"], justify="center")

    def save_caption(self):
        idx = self._current_index()
        if idx is None:
            messagebox.showinfo(APP_TITLE, "Select an image from the list.")
            return
        job = self.jobs[idx]
        text = self.txt_caption.get("1.0", "end").strip()
        out = job.path.with_suffix(self._collect_settings().extension)
        out.write_text(text + "\n", encoding="utf-8")
        job.caption = text
        if job.status != "error":
            job.status = "ok"
        self._refresh_row(idx)
        self.lbl_caption_file.configure(text=out.name + " (exists)")
        self._log(f"💾 Saved {out.name}.")

    # ---- backend / models ------------------------------------------------ #
    @property
    def backend(self) -> str:
        label = self.var_backend_label.get()
        return next((b for b, l in BACKEND_LABELS.items() if l == label), "ollama")

    def _on_backend_change(self, initial: bool = False):
        b = self.backend
        state = "normal" if b == "ollama" else "disabled"
        self.ent_url.configure(state=state)
        self.btn_models.configure(state="normal" if b == "llamacpp" else "disabled")
        self.var_model.set(self._models_by_backend.get(b, ""))
        self.cmb_model.bind("<<ComboboxSelected>>", lambda e: self._models_by_backend.__setitem__(self.backend, self.var_model.get()))
        if not initial:
            self.refresh_models()

    def _lc_registry(self) -> ModelRegistry:
        from captionz_llamacpp import DEFAULT_DIR
        return ModelRegistry(Path(self.settings.llamacpp_dir) if self.settings.llamacpp_dir else DEFAULT_DIR)

    def refresh_models(self):
        b = self.backend
        url = self.var_url.get().strip() or DEFAULT_OLLAMA_URL
        self.lbl_conn.configure(text="…")

        def work():
            try:
                if b == "ollama":
                    models = OllamaClient(url, timeout=15).list_vision_models(self.settings.vision_blocklist)
                elif b == "llamacpp":
                    models = self._lc_registry().list() or [LC_DEFAULT["name"]]
                else:
                    from captionz_hf import HF_MODELS
                    models = list(HF_MODELS)
                self.ui_queue.put(("models", (b, models), None))
            except Exception as e:  # noqa: BLE001
                self.ui_queue.put(("models", (b, []), str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _apply_models(self, payload, error: str | None):
        b, models = payload
        if b != self.backend:
            return
        if error:
            self.lbl_conn.configure(text="✖ offline" if b == "ollama" else "✖ error")
            self._log(f"{'Could not connect to Ollama' if b == 'ollama' else 'Error'}: {error}")
            self.cmb_model["values"] = []
            return
        self.cmb_model["values"] = models
        if models:
            if self.var_model.get() not in models:
                self.var_model.set(models[0])
            self._models_by_backend[b] = self.var_model.get()
            if b == "llamacpp":
                have = self._lc_registry().list()
                self.lbl_conn.configure(text=f"✔ {len(have)} local model(s)" if have
                                        else "Default model will be downloaded on first launch (about 2.9 GB)")
            else:
                self.lbl_conn.configure(text=f"✔ {len(models)} vision model(s)")
        else:
            self.lbl_conn.configure(text="No vision models found")
            if b == "ollama":
                self._log("No vision models found. Example: `ollama pull qwen3-vl:8b`.")

    # ---- llama.cpp model manager ----------------------------------------- #
    def open_model_manager(self):
        reg = self._lc_registry()
        win = tk.Toplevel(self)
        win.title("llama.cpp models (no Ollama)")
        win.geometry("720x460")
        win.transient(self)
        pad = {"padx": 6, "pady": 4}
        ttk.Label(win, text=f"Folder: {reg.models_dir}").pack(anchor="w", **pad)
        lb = tk.Listbox(win, height=8)
        lb.pack(fill="both", expand=True, padx=6)
        t = getattr(self, "_theme", THEMES["light"])
        lb.configure(bg=t["field"], fg=t["fg"], selectbackground=t["sel"])

        def fill():
            lb.delete(0, "end")
            for m in reg.list():
                info = reg.info(m)
                lb.insert("end", f"{m}   [{info.get('source', '?')}]  {info.get('repo') or info.get('ollama_name') or ''}")
            self.refresh_models()

        def run(label, fn):
            def worker():
                self.ui_queue.put(("log", f"▶ {label}…"))
                try:
                    fn(lambda m: self.ui_queue.put(("log", m)))
                    self.ui_queue.put(("log", f"✔ {label} finished"))
                except Exception as e:  # noqa: BLE001
                    self.ui_queue.put(("log", f"✖ {label} : {e}"))
                self.after(0, fill)
            threading.Thread(target=worker, daemon=True).start()

        row1 = ttk.Frame(win); row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Download from Hugging Face:").pack(side="left")
        var_known = tk.StringVar(value=LC_DEFAULT["name"])
        ttk.Combobox(row1, textvariable=var_known, state="readonly", width=34, values=list(LC_KNOWN)).pack(side="left", padx=4)
        ttk.Button(row1, text="Download", command=lambda: run(
            f"Downloading {var_known.get()}", lambda log: reg.add_known(var_known.get(), log))).pack(side="left")

        row2 = ttk.Frame(win); row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Import from Ollama:").pack(side="left")
        var_oll = tk.StringVar()
        cmb_oll = ttk.Combobox(row2, textvariable=var_oll, state="readonly", width=44)
        cmb_oll.pack(side="left", padx=4)
        try:
            cmb_oll["values"] = reg.ollama_vision_models()
            if cmb_oll["values"]:
                var_oll.set(cmb_oll["values"][0])
        except Exception:
            pass
        ttk.Button(row2, text="Import", command=lambda: var_oll.get() and run(
            f"Importing {var_oll.get()}", lambda log: reg.import_from_ollama(var_oll.get(), None, log))).pack(side="left")

        row3 = ttk.Frame(win); row3.pack(fill="x", **pad)
        def selected():
            sel = lb.curselection()
            return lb.get(sel[0]).split("   ")[0] if sel else ""
        ttk.Button(row3, text="Use", command=lambda: (self._models_by_backend.__setitem__("llamacpp", selected()),
                                                            self.var_model.set(selected()))).pack(side="left")
        ttk.Button(row3, text="Update (Hugging Face)", command=lambda: selected() and run(
            f"Updating {selected()}", lambda log: reg.update_from_source(selected(), log))).pack(side="left", padx=4)
        ttk.Button(row3, text="Remove", command=lambda: selected() and messagebox.askyesno(
            APP_TITLE, f"Remove {selected()}?") and (reg.remove(selected()), fill())).pack(side="left", padx=4)
        ttk.Separator(row3, orient="vertical").pack(side="left", fill="y", padx=8)
        binary = ServerBinary(reg.models_dir.parent, self.settings.llamacpp_build)
        cur = binary.current()
        ttk.Label(row3, text=f"llama-server: {cur['tag'] + ' ' + cur['build'] if cur else 'not installed (automatic on first launch)'}").pack(side="left")
        ttk.Button(row3, text="Update llama-server", command=lambda: run(
            "Updating llama-server", lambda log: binary.update(log))).pack(side="left", padx=4)
        ttk.Label(win, text="Downloads and imports appear in the main window log.",
                  foreground=t["muted"]).pack(anchor="w", **pad)
        fill()

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
            self.tree.insert("", "end", iid=str(idx), values=(str(img), "pending", ""))
            added += 1
        self._update_count()
        self._log(f"{added} image(s) added ({len(images) - added} duplicate(s) skipped).")
        if added and not self.tree.selection():
            first = str(len(self.jobs) - added)
            self.tree.selection_set(first)
            self.tree.see(first)

    def add_file(self):
        f = filedialog.askopenfilename(title="Choose an image", filetypes=self._filetypes())
        if f:
            self._add_paths([Path(f)])

    def add_files(self):
        fs = filedialog.askopenfilenames(title="Choose images", filetypes=self._filetypes())
        if fs:
            self._add_paths([Path(f) for f in fs])

    def add_folder(self):
        d = filedialog.askdirectory(title="Choose an image folder")
        if d:
            self._add_paths([Path(d)])

    @staticmethod
    def _filetypes():
        pat = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        return [("Images", pat), ("All files", "*.*")]

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
                self._log(f"Could not read clipboard: {e}")
        if isinstance(data, list):                       # files copied in the explorer
            paths = [Path(p) for p in data]
        elif data is not None and Image is not None and isinstance(data, Image.Image):
            out = save_pasted_image(data, self.settings.paste_path)
            paths = [out]
            self._log(f"Pasted image saved: {out}")
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
            msg = "No image found in the clipboard."
            if ImageGrab is None:
                msg += " Install Pillow (pip install pillow) to paste screenshots."
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
            messagebox.showinfo(APP_TITLE, "Select one or more images from the list.")
            return
        self.start(sel)

    def start_current(self):
        idx = self._current_index()
        if idx is None:
            messagebox.showinfo(APP_TITLE, "Select an image from the list.")
            return
        self.start([idx], force=True)  # explicitly requested: always overwrite

    def start(self, indices: list[int] | None, force: bool = False):
        if self.captioner.is_running():
            return
        s = self._collect_settings()
        if not active_model(s) and s.backend == "ollama":
            messagebox.showwarning(APP_TITLE, "Select a vision model.")
            return
        if not self.jobs:
            messagebox.showwarning(APP_TITLE, "Add at least one image or folder.")
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
        self.lbl_status.configure(text="Starting…")
        self._log(f"Starting: {len(indices)} image(s) · {BACKEND_LABELS[s.backend]} · “{active_model(s) or 'default'}”"
                  f"{'' if Image else ' (Pillow unavailable: sending original images)'}.")
        self.captioner.start(self.jobs, indices, s, force)

    def stop(self):
        if self.captioner.is_running():
            self.captioner.stop()
            self._log("Stop requested; waiting for the current image to finish…")

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
                    if job.status == "error":
                        self._log(f"✖ {job.path.name}: {job.error}")
                    elif job.status == "ok" and idx == self._current_index():
                        self.txt_caption.delete("1.0", "end")
                        self.txt_caption.insert("1.0", job.caption)
                        self.lbl_caption_file.configure(
                            text=job.path.with_suffix(self.settings.extension).name + " (exists)")
                elif ev[0] == "phase":
                    if ev[2] == "loading":
                        self._log(f"Loading model “{active_model(self.settings) or 'default'}”…")
                elif ev[0] == "log":
                    self._log(ev[1])
                elif ev[0] == "done":
                    self._on_done()
        except queue.Empty:
            pass
        try:
            while True:
                kind, a, b = self.ui_queue.get_nowait()
                if kind == "log":
                    self._log(a)
        except queue.Empty:
            pass
        if self.captioner.is_running():  # live status: phase, timer, %, ETA
            snap = self.captioner.progress.snapshot()
            self.progress.configure(value=int(snap["fraction"] * 1000))
            self.lbl_progress.configure(text=f"{snap['fraction'] * 100:.0f}%")
            self.lbl_status.configure(text=snap["text"])
            for i, j in enumerate(self.jobs):
                if j.status == "processing":
                    self._refresh_row(i)
        self.after(100, self._poll_ui_queue)

    def _on_done(self):
        ok = sum(j.status == "ok" for j in self.jobs)
        err = sum(j.status == "error" for j in self.jobs)
        skip = sum(j.status == "skipped" for j in self.jobs)
        stopped = self.captioner.stop_event.is_set()
        snap = self.captioner.progress.snapshot()
        self.progress.configure(value=int(snap["fraction"] * 1000))
        self.lbl_progress.configure(text=f"{snap['fraction'] * 100:.0f}%")
        self.lbl_status.configure(text=snap["text"])
        self._log(f"{'Stopped' if stopped else 'Finished'}: {ok} successful, {skip} skipped, {err} error(s) · {snap['text']}")
        for b in (self.btn_start, self.btn_sel):
            b.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _on_close(self):
        if self.captioner.is_running():
            if not messagebox.askyesno(APP_TITLE, "A task is running. Quit anyway?"):
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
    ap = argparse.ArgumentParser(description="Captionz — image captioning with Ollama vision models")
    ap.add_argument("--ui", choices=["tk", "web"], default="tk",
                    help="tk = Tkinter desktop UI (default), web = NiceGUI web UI")
    ap.add_argument("--port", type=int, default=8080, help="web UI port (default: 8080)")
    ap.add_argument("--host", default="127.0.0.1", help="web UI host (default: 127.0.0.1; use 0.0.0.0 to expose it on the LAN)")
    ap.add_argument("--no-browser", action="store_true", help="web UI: do not open a browser automatically")
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
