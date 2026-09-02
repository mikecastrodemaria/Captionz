"""
Captionz core — shared by the Tkinter UI (app.py) and the NiceGUI web UI (webui.py).

Contains everything that is independent of the UI toolkit:
  - OllamaClient: /api/tags, /api/show, /api/chat with images, vision detection
    (crispz-studio patterns: capability check with name fallback, JPEG
    downscaling, <think> stripping, keep_alive / CPU options)
  - Prompt composition: caption type × length × options × {name}, or custom prompt
  - Settings (persisted in settings.json), Job, collect_images
  - Captioner: runs a list of jobs in a background thread and reports through a queue
"""

from __future__ import annotations

import base64
import io
import json
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image  # optional: preview + downscaling before upload
except ImportError:  # pragma: no cover
    Image = None

APP_TITLE = "Captionz · Ollama vision captioning"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "settings.json"

# --------------------------------------------------------------------------- #
# Prompt composition (JoyCaption-style: type × length × options)
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
DEFAULT_OPTIONS = ["Output only the caption, no preamble, no quotes, no markdown."]


def build_prompt(caption_type: str, length: str, options: list[str], name: str, custom: str) -> str:
    """Final prompt sent to the model. A custom prompt overrides everything
    ({name} is substituted there too)."""
    name = name.strip() or "the main character"
    if custom.strip():
        return custom.strip().replace("{name}", name)
    template = CAPTION_TYPES.get(caption_type, CAPTION_TYPES["Descriptive (formal)"])
    base = template.replace("{length}", CAPTION_LENGTHS.get(length, ""))
    base = re.sub(r"\s{2,}", " ", base).replace(" .", ".").replace("Be .", "").strip()
    parts = [base] + [o.replace("{name}", name) for o in options]
    return " ".join(p.strip() for p in parts if p.strip())


DEFAULT_PROMPT = build_prompt("Training caption (paragraph)", "long", DEFAULT_OPTIONS, "", "")
# Clearly multimodal names: fallback when /api/show does not return `capabilities`
VISION_NAME_HINTS = ("llava", "-vl", "vl:", "moondream", "minicpm-v", "bakllava",
                     "llama3.2-vision", "llama-3.2-vision", "vision")
_THINK_RE = re.compile(r"<think>[\s\S]*?(?:</think>|$)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Ollama client (stdlib only)
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
        """Models that can really do vision (crispz-studio pattern).

        Authoritative source: the "vision" capability from /api/show (or from
        /api/tags on recent Ollama versions). If the field is missing (old
        version), fall back to a clearly multimodal name. Families (clip…) are
        NOT used: they give false positives.
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
        """Image -> base64. Downscaled to JPEG when Pillow is available (fewer
        tokens, faster upload, same captions); raw bytes otherwise."""
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
        """"Thinking" models (Qwen3+, DeepSeek-R1…): the reasoning must never end
        up in the caption. Keep what follows the last </think>."""
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        return _THINK_RE.sub("", text).strip().strip('"')

    def caption(self, model: str, prompt: str, image_path: Path, temperature: float = 0.2,
                keep_alive: str | int = "10m", max_side: int = 1024, cpu_only: bool = False) -> str:
        options: dict = {"temperature": temperature}
        if cpu_only:
            options["num_gpu"] = 0  # no shared VRAM (slower)
        payload = {
            "model": model, "stream": False, "keep_alive": keep_alive, "options": options,
            "messages": [{"role": "user", "content": prompt,
                          "images": [self.encode_image(image_path, max_side)]}],
        }
        resp = self._request("POST", "/api/chat", payload)
        return self.strip_thinking((resp.get("message") or {}).get("content", ""))


# --------------------------------------------------------------------------- #
# Data model
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
    backend: str = "ollama"         # ollama | hf (transformers, Hugging Face Spaces)
    ollama_url: str = DEFAULT_OLLAMA_URL
    model: str = ""
    hf_model: str = ""              # transformers model id when backend == "hf"
    caption_type: str = "Training caption (paragraph)"
    caption_length: str = "long"
    options: list = field(default_factory=lambda: list(DEFAULT_OPTIONS))
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
    paste_dir: str = ""              # folder for pasted images (empty = ./pasted)
    vision_blocklist: list = field(default_factory=list)

    @property
    def prompt(self) -> str:
        return build_prompt(self.caption_type, self.caption_length, self.options, self.name, self.custom_prompt)

    @property
    def paste_path(self) -> Path:
        return Path(self.paste_dir) if self.paste_dir else APP_DIR / "pasted"

    def normalized(self) -> "Settings":
        if not self.extension.startswith("."):
            self.extension = "." + self.extension
        ka = str(self.keep_alive).strip() or "0"
        self.keep_alive = int(ka) if ka.isdigit() else ka  # type: ignore[assignment]
        return self

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


def save_pasted_image(img, paste_dir: Path) -> Path:
    """Save a PIL image from the clipboard / browser paste as PNG, unique name."""
    paste_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("paste_%Y%m%d_%H%M%S")
    out = paste_dir / f"{stamp}.png"
    n = 1
    while out.exists():
        out = paste_dir / f"{stamp}_{n}.png"
        n += 1
    img.convert("RGB").save(out, "PNG")
    return out


# --------------------------------------------------------------------------- #
# Backends: where captions come from (Ollama locally, transformers on Spaces)
# --------------------------------------------------------------------------- #
class Backend:
    """Minimal interface every backend implements."""
    name = "base"

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def caption(self, model: str, prompt: str, image_path: Path, *, temperature: float = 0.2,
                max_side: int = 1024) -> str:
        raise NotImplementedError


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, url: str = DEFAULT_OLLAMA_URL, keep_alive: str | int = "10m",
                 cpu_only: bool = False, blocklist: list[str] | None = None):
        self.client = OllamaClient(url)
        self.keep_alive = keep_alive
        self.cpu_only = cpu_only
        self.blocklist = blocklist or []

    def list_models(self) -> list[str]:
        return OllamaClient(self.client.base_url, timeout=15).list_vision_models(self.blocklist)

    def caption(self, model, prompt, image_path, *, temperature=0.2, max_side=1024) -> str:
        return self.client.caption(model, prompt, image_path, temperature=temperature,
                                   keep_alive=self.keep_alive, max_side=max_side, cpu_only=self.cpu_only)


BACKENDS = ("ollama", "hf")


def make_backend(s: "Settings") -> Backend:
    """Backend chosen by settings: "ollama" (default) or "hf" (transformers,
    see captionz_hf.py — used on Hugging Face Spaces)."""
    if s.backend == "hf":
        from captionz_hf import HFBackend  # lazy: torch/transformers are heavy and optional
        return HFBackend(s.hf_model or None)
    return OllamaBackend(s.ollama_url, s.keep_alive, s.cpu_only, s.vision_blocklist)


# --------------------------------------------------------------------------- #
# Captioning logic (UI-independent, used by CLI, Tkinter, NiceGUI, Gradio)
# --------------------------------------------------------------------------- #
def caption_job(job: Job, s: "Settings", backend: Backend, force: bool = False) -> Job:
    """Caption one image and write the text file next to it, honouring the
    skip / overwrite / append policy, prefix/suffix and single-line options.
    Updates and returns the job (status: ok | ignoré | erreur)."""
    out = job.path.with_suffix(s.extension)
    if out.exists() and s.existing == "skip" and not force:
        job.status, job.error = "ignoré", "caption déjà présente"
        return job
    job.status = "en cours"
    t0 = time.time()
    try:
        text = backend.caption(s.model, s.prompt, job.path, temperature=s.temperature, max_side=s.max_side)
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
    return job


def run_jobs(jobs: list[Job], indices: list[int] | None, s: "Settings", force: bool = False,
             stop_event: threading.Event | None = None, backend: Backend | None = None):
    """Generator over a batch. Yields ("row", idx) when a job starts and when it
    ends, then ("progress", done, total). Stops early when stop_event is set."""
    backend = backend or make_backend(s)
    if indices is None:
        indices = list(range(len(jobs)))
    total = len(indices)
    for n, idx in enumerate(indices, 1):
        if stop_event is not None and stop_event.is_set():
            break
        job = jobs[idx]
        out = job.path.with_suffix(s.extension)
        if not (out.exists() and s.existing == "skip" and not force):
            job.status = "en cours"
            yield ("row", idx)
        caption_job(job, s, backend, force)
        yield ("row", idx)
        yield ("progress", n, total)


class Captioner:
    """Runs run_jobs() in a thread for the GUIs. Events are pushed to `events`
    as tuples: ("row", idx), ("progress", done, total), ("done",)."""

    def __init__(self) -> None:
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def start(self, jobs: list[Job], indices: list[int], s: "Settings", force: bool = False) -> None:
        if self.is_running():
            return
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._run, args=(jobs, indices, s, force), daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self, jobs: list[Job], indices: list[int], s: "Settings", force: bool) -> None:
        try:
            for ev in run_jobs(jobs, indices, s, force, self.stop_event):
                self.events.put(ev)
        except Exception as e:  # noqa: BLE001  (e.g. backend import failure)
            for idx in indices:
                if jobs[idx].status in ("en attente", "en cours"):
                    jobs[idx].status, jobs[idx].error = "erreur", str(e)
                    self.events.put(("row", idx))
        self.events.put(("done",))
