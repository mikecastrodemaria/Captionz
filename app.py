"""
Captionz — batch image captioning through Ollama vision models.

The Ollama layer reuses patterns from crispz-studio (cz_ollama.py): vision
detection through /api/show with a name-based fallback, images downscaled to
JPEG before upload (when Pillow is installed), stripping of <think> blocks,
configurable keep_alive / CPU mode so the model does not hog VRAM.

Features:
  - Connect to an Ollama server (configurable URL), models filtered on "vision"
  - Sources: a single file, a selection of files, or a folder (recursive or not)
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

import base64
import io
import json
import queue
import re
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from PIL import Image, ImageTk  # optionnel : aperçu + réduction des images
except ImportError:  # pragma: no cover
    Image = ImageTk = None

APP_TITLE = "Captionz · Ollama vision captioning"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
SETTINGS_FILE = Path(__file__).with_name("settings.json")

# --------------------------------------------------------------------------- #
# Composition du prompt (inspiré de JoyCaption : type × longueur × options)
# --------------------------------------------------------------------------- #
CAPTION_TYPES: dict[str, str] = {
    "Descriptive (formal)":
        "Write a {length} descriptive caption for this image in a formal tone.",
    "Descriptive (casual)":
        "Write a {length} descriptive caption for this image in a casual, natural tone.",
    "Training caption (paragraph)":
        "Write a {length} caption of this image for use as a training caption. Mention the subject, "
        "setting, composition, lighting, colors, style and mood. Do not start with 'This image shows'.",
    "Stable Diffusion prompt":
        "Output a {length} Stable Diffusion prompt that would generate this image: comma-separated "
        "visual tags (subject, clothing, setting, lighting, style, quality).",
    "Booru tag list":
        "Write a {length} comma-separated list of Booru-style tags for this image, lowercase, "
        "underscores instead of spaces.",
    "Art critic analysis":
        "Analyze this image like an art critic would, with information about its composition, style, "
        "symbolism, the use of color, light, any artistic movement it belongs to, etc. Be {length}.",
    "Product listing":
        "Write a {length} product listing style caption for this image.",
    "Social media post":
        "Write a {length} caption for this image as if it were a social media post.",
    "Short sentence":
        "Describe this image in one short sentence.",
}
CAPTION_LENGTHS: dict[str, str] = {
    "any": "",
    "very short": "very short",
    "short": "short",
    "medium-length": "medium-length",
    "long": "long",
    "very long": "very long",
}
EXTRA_OPTIONS: list[str] = [
    "If there is a person/character in the image you must refer to them as {name}.",
    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), "
    "but do still include changeable attributes (like hair style).",
    "Include information about lighting.",
    "Include information about camera angle.",
    "Include information about whether there is a watermark or not.",
    "Include information about whether there are JPEG artifacts or not.",
    "If it is a photo you MUST include information about what camera was likely used and details such as "
    "aperture, shutter speed, ISO, etc.",
    "Do NOT include anything sexual; keep it PG.",
    "Do NOT mention the image's resolution.",
    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
    "Do NOT mention any text that is in the image.",
    "Specify the depth of field and whether the background is in focus or blurred.",
    "If applicable, mention the likely use of artificial or natural lighting sources.",
    "Do NOT use any ambiguous language.",
    "Include whether the image is sfw, suggestive, or nsfw.",
    "ONLY describe the most important elements of the image.",
    "Avoid any word that would be blocked by Midjourney, DALL-E or similar content filters (nudity, sexual terms, "
    "gore, blood, violence, weapons, drugs, body parts, real celebrity names, political or religious figures); "
    "rephrase with neutral, safe wording instead of omitting the element.",
    "Do NOT start with 'This image shows' or 'The image depicts'.",
    "Output only the caption, no preamble, no quotes, no markdown.",
]


def build_prompt(caption_type: str, length: str, options: list[str], name: str, custom: str) -> str:
    """Prompt final envoyé au modèle. Un prompt personnalisé remplace tout
    (mais {name} y est aussi substitué)."""
    name = name.strip() or "the main character"
    if custom.strip():
        return custom.strip().replace("{name}", name)
    template = CAPTION_TYPES.get(caption_type, CAPTION_TYPES["Descriptive (formal)"])
    length_word = CAPTION_LENGTHS.get(length, "")
    base = template.replace("{length}", length_word)
    base = re.sub(r"\s{2,}", " ", base).replace(" .", ".").replace("Be .", "").strip()
    parts = [base] + [o.replace("{name}", name) for o in options]
    return " ".join(p.strip() for p in parts if p.strip())


DEFAULT_PROMPT = build_prompt("Training caption (paragraph)", "long",
                              ["Output only the caption, no preamble, no quotes, no markdown."], "", "")
# Noms clairement multimodaux : repli quand /api/show ne renvoie pas `capabilities`
VISION_NAME_HINTS = ("llava", "-vl", "vl:", "moondream", "minicpm-v", "bakllava",
                     "llama3.2-vision", "llama-3.2-vision", "vision")
_THINK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Client Ollama (stdlib uniquement)
# --------------------------------------------------------------------------- #
class OllamaClient:
    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self) -> list[dict]:
        return self._request("GET", "/api/tags").get("models", [])

    def show(self, name: str) -> dict:
        return self._request("POST", "/api/show", {"model": name})

    def list_vision_models(self, blocklist: list[str] | None = None) -> list[str]:
        """Modèles réellement capables de vision (pattern crispz-studio).

        Source autoritaire : la capacité "vision" de /api/show (ou de /api/tags
        sur les Ollama récents). Si le champ est absent (vieille version), repli
        sur un nom clairement multimodal. Les familles (clip…) ne sont PAS
        utilisées : elles donnent des faux positifs.
        """
        block = [b.lower() for b in (blocklist or []) if b]
        vision = []
        for m in self.list_models():
            name = m.get("name") or m.get("model")
            if not name or any(b in name.lower() for b in block):
                continue
            caps = m.get("capabilities")
            if caps is None:
                try:
                    caps = self.show(name).get("capabilities")
                except Exception:
                    caps = None
            if caps:
                if "vision" in [c.lower() for c in caps]:
                    vision.append(name)
            elif any(k in name.lower() for k in VISION_NAME_HINTS):
                vision.append(name)
        vision.sort(key=lambda n: (0 if any(k in n.lower() for k in VISION_NAME_HINTS) else 1, n.lower()))
        return vision

    @staticmethod
    def encode_image(path: Path, max_side: int = 1024, quality: int = 85) -> str:
        """Image -> base64. Réduite en JPEG si Pillow est dispo (moins de tokens,
        upload plus rapide, mêmes captions) ; sinon octets bruts."""
        if Image is not None and max_side > 0:
            try:
                img = Image.open(path).convert("RGB")
                w, h = img.size
                if max(w, h) > max_side:
                    if w >= h:
                        img = img.resize((max_side, int(h * max_side / w)), Image.LANCZOS)
                    else:
                        img = img.resize((int(w * max_side / h), max_side), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=quality)
                return base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception:
                pass
        return base64.b64encode(path.read_bytes()).decode("ascii")

    @staticmethod
    def strip_thinking(text: str) -> str:
        """Modèles "thinking" (Qwen3+, DeepSeek-R1…) : le raisonnement ne doit
        jamais finir dans la caption. On garde ce qui suit le dernier </think>."""
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        return _THINK_RE.sub("", text).strip().strip('"')

    def caption(self, model: str, prompt: str, image_path: Path, temperature: float = 0.2,
                keep_alive: str | int = "10m", max_side: int = 1024, cpu_only: bool = False) -> str:
        options: dict = {"temperature": temperature}
        if cpu_only:
            options["num_gpu"] = 0  # 0 VRAM partagée (plus lent)
        payload = {
            "model": model, "stream": False, "keep_alive": keep_alive, "options": options,
            "messages": [{"role": "user", "content": prompt,
                          "images": [self.encode_image(image_path, max_side)]}],
        }
        resp = self._request("POST", "/api/chat", payload)
        return self.strip_thinking((resp.get("message") or {}).get("content", ""))


# --------------------------------------------------------------------------- #
# Modèle de données
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    path: Path
    status: str = "en attente"
    caption: str = ""
    error: str = ""
    duration: float = 0.0


@dataclass
class Settings:
    ollama_url: str = DEFAULT_OLLAMA_URL
    model: str = ""
    caption_type: str = "Training caption (paragraph)"
    caption_length: str = "long"
    options: list = field(default_factory=lambda: ["Output only the caption, no preamble, no quotes, no markdown."])
    name: str = ""
    custom_prompt: str = ""
    prefix: str = ""
    suffix: str = ""
    extension: str = ".txt"
    recursive: bool = True
    existing: str = "skip"          # skip | overwrite | append
    temperature: float = 0.2
    single_line: bool = True
    keep_alive: str = "10m"
    max_side: int = 1024
    cpu_only: bool = False
    dark: bool = False
    vision_blocklist: list = field(default_factory=list)

    @property
    def prompt(self) -> str:
        return build_prompt(self.caption_type, self.caption_length, self.options, self.name, self.custom_prompt)

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(SETTINGS_FILE.read_text("utf-8"))
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception:
            return cls()

    def save(self) -> None:
        try:
            SETTINGS_FILE.write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False), "utf-8")
        except Exception:
            pass


def collect_images(paths: list[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found.extend(f for f in it if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            found.append(p)
    seen: set[Path] = set()
    out = []
    for f in sorted(found):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# --------------------------------------------------------------------------- #
# Thèmes
# --------------------------------------------------------------------------- #
THEMES = {
    "light": dict(bg="#f3f3f3", fg="#1b1b1b", field="#ffffff", sel="#cfe3ff", border="#c8c8c8",
                  ok="#1a7f37", err="#c62828", skip="#8a6d00", run="#0b57d0", muted="#666666"),
    "dark": dict(bg="#1e1f22", fg="#e6e6e6", field="#2b2d31", sel="#3b4b66", border="#3c3f44",
                 ok="#5ed08a", err="#ff6b6b", skip="#e0c060", run="#7fb3ff", muted="#9a9a9a"),
}


class ScrollFrame(ttk.Frame):
    """Frame défilable verticalement (pour la longue liste d'options)."""

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
# Interface
# --------------------------------------------------------------------------- #
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1480x920")
        self.minsize(1100, 700)

        self.settings = Settings.load()
        self.jobs: list[Job] = []
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self._preview_img = None
        self._text_widgets: list[tk.Text] = []

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

        # ================= colonne gauche : réglages =================
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
        self.var_recursive = tk.BooleanVar(value=s.recursive)
        ttk.Checkbutton(src, text="récursif", variable=self.var_recursive).pack(side="left", **pad)
        ttk.Separator(src, orient="vertical").pack(side="left", fill="y", padx=8, pady=4)
        ttk.Button(src, text="Retirer sélection", command=self.remove_selected).pack(side="left", **pad)
        ttk.Button(src, text="Vider", command=self.clear_jobs).pack(side="left", **pad)
        self.lbl_count = ttk.Label(src, text="0 image")
        self.lbl_count.pack(side="right", **pad)

        # --- prompt composé ---
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

        # --- sortie ---
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

        # --- modèle / perf ---
        mf = ttk.LabelFrame(left, text="Modèle")
        mf.pack(fill="x", **pad)
        self.var_temp = tk.DoubleVar(value=s.temperature)
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

        # --- contrôles ---
        ctl = ttk.Frame(left)
        ctl.pack(fill="x", **pad)
        self.btn_start = ttk.Button(ctl, text="▶ Captionner tout", command=lambda: self.start(None))
        self.btn_start.pack(side="left", **pad)
        self.btn_sel = ttk.Button(ctl, text="▶ Captionner la sélection", command=self.start_selected)
        self.btn_sel.pack(side="left", **pad)
        self.btn_stop = ttk.Button(ctl, text="■ Arrêter", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", **pad)
        ttk.Button(ctl, text="🌓 Mode sombre", command=self.toggle_theme).pack(side="right", **pad)
        self.progress = ttk.Progressbar(ctl, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, **pad)
        self.lbl_progress = ttk.Label(ctl, text="")
        self.lbl_progress.pack(side="left", **pad)

        self.log = scrolledtext.ScrolledText(left, height=5, state="disabled", wrap="word")
        self.log.pack(fill="x", padx=6, pady=(0, 6))
        self._text_widgets.append(self.log)

        # ================= colonne droite : images =================
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

    # ---- thème ------------------------------------------------------------ #
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
        for sf in self._all_children(self, tk.Canvas):
            if sf is not self.canvas:
                sf.configure(bg=t["bg"])
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

    # ---- helpers UI ----------------------------------------------------- #
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
        s = Settings(
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
            single_line=self.var_single.get(),
            keep_alive=self.var_keep.get().strip() or "0",
            max_side=int(self.var_maxside.get() or 0),
            cpu_only=self.var_cpu.get(),
            dark=self.settings.dark,
            vision_blocklist=list(self.settings.vision_blocklist),
        )
        if not s.extension.startswith("."):
            s.extension = "." + s.extension
        if s.keep_alive.isdigit():
            s.keep_alive = int(s.keep_alive)  # type: ignore[assignment]
        return s

    def _update_count(self):
        n = len(self.jobs)
        self.lbl_count.configure(text=f"{n} image{'s' if n > 1 else ''}")

    def _refresh_row(self, idx: int):
        job = self.jobs[idx]
        tag = {"ok": "ok", "erreur": "err", "ignoré": "skip", "en cours": "run"}.get(job.status, "")
        dur = f"{job.duration:.1f}s" if job.duration else ""
        self.tree.item(str(idx), values=(str(job.path), job.status, dur), tags=(tag,))

    def _selected_indices(self) -> list[int]:
        return [int(i) for i in self.tree.selection()]

    def _current_index(self) -> int | None:
        sel = self._selected_indices()
        return sel[0] if sel else None

    # ---- aperçu + caption éditable ---------------------------------------- #
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
        path = getattr(self, "_preview_path", None)
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
                img = tk.PhotoImage(file=str(path))  # PNG/GIF seulement
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

    # ---- modèles -------------------------------------------------------- #
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

    # ---- sources -------------------------------------------------------- #
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

    def remove_selected(self):
        if self._is_running():
            return
        sel = set(self._selected_indices())
        if not sel:
            return
        self.jobs = [j for i, j in enumerate(self.jobs) if i not in sel]
        self._rebuild_tree()

    def clear_jobs(self):
        if self._is_running():
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

    # ---- exécution ------------------------------------------------------ #
    def _is_running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

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
        # une image explicitement demandée : on écrase toujours
        self.start([idx], force=True)

    def start(self, indices: list[int] | None, force: bool = False):
        if self._is_running():
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
        self.stop_event.clear()
        for b in (self.btn_start, self.btn_sel):
            b.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress.configure(maximum=len(indices), value=0)
        self.lbl_progress.configure(text=f"0/{len(indices)}")
        self._log(f"Démarrage : {len(indices)} image(s) avec « {s.model} »"
                  f"{'' if Image else ' (Pillow absent : images envoyées brutes)'}.")
        self.worker = threading.Thread(target=self._run, args=(s, indices, force), daemon=True)
        self.worker.start()

    def stop(self):
        if self._is_running():
            self.stop_event.set()
            self._log("Arrêt demandé, fin de l'image en cours…")

    def _run(self, s: Settings, indices: list[int], force: bool):
        client = OllamaClient(s.ollama_url)
        prompt = s.prompt
        for n, idx in enumerate(indices, 1):
            if self.stop_event.is_set():
                break
            job = self.jobs[idx]
            out = job.path.with_suffix(s.extension)
            if out.exists() and s.existing == "skip" and not force:
                job.status, job.error = "ignoré", "caption déjà présente"
                self.ui_queue.put(("row", idx, None))
                self.ui_queue.put(("progress", n, len(indices)))
                continue
            job.status = "en cours"
            self.ui_queue.put(("row", idx, None))
            t0 = time.time()
            try:
                text = client.caption(s.model, prompt, job.path, temperature=s.temperature,
                                      keep_alive=s.keep_alive, max_side=s.max_side, cpu_only=s.cpu_only)
                if s.single_line:
                    text = " ".join(text.split())
                text = f"{s.prefix}{text}{s.suffix}".strip()
                if not text:
                    raise RuntimeError("réponse vide du modèle")
                if out.exists() and s.existing == "append" and not force:
                    old = out.read_text("utf-8").rstrip("\n")
                    text = (old + "\n" + text) if old else text
                out.write_text(text + "\n", encoding="utf-8")
                job.caption, job.status, job.error = text, "ok", ""
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:300]
                job.status, job.error = "erreur", f"HTTP {e.code}: {body}"
            except Exception as e:  # noqa: BLE001
                job.status, job.error = "erreur", str(e)
            job.duration = time.time() - t0
            self.ui_queue.put(("row", idx, None))
            self.ui_queue.put(("progress", n, len(indices)))
        self.ui_queue.put(("done", None, None))

    # ---- boucle UI ------------------------------------------------------ #
    def _poll_ui_queue(self):
        try:
            while True:
                kind, a, b = self.ui_queue.get_nowait()
                if kind == "models":
                    self._apply_models(a, b)
                elif kind == "row":
                    self._refresh_row(a)
                    job = self.jobs[a]
                    if job.status == "erreur":
                        self._log(f"✖ {job.path.name} : {job.error}")
                    elif job.status == "ok" and a == self._current_index():
                        self.txt_caption.delete("1.0", "end")
                        self.txt_caption.insert("1.0", job.caption)
                        self.lbl_caption_file.configure(
                            text=job.path.with_suffix(self.settings.extension).name + " (existe)")
                elif kind == "progress":
                    self.progress.configure(value=a)
                    self.lbl_progress.configure(text=f"{a}/{b}")
                elif kind == "done":
                    self._on_done()
        except queue.Empty:
            pass
        self.after(100, self._poll_ui_queue)

    def _on_done(self):
        ok = sum(j.status == "ok" for j in self.jobs)
        err = sum(j.status == "erreur" for j in self.jobs)
        skip = sum(j.status == "ignoré" for j in self.jobs)
        self._log(f"{'Arrêté' if self.stop_event.is_set() else 'Terminé'} : {ok} ok, {skip} ignoré(s), {err} erreur(s).")
        for b in (self.btn_start, self.btn_sel):
            b.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _on_close(self):
        if self._is_running():
            if not messagebox.askyesno(APP_TITLE, "Un traitement est en cours. Quitter quand même ?"):
                return
            self.stop_event.set()
        try:
            self._collect_settings().save()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
