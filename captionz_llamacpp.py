"""
Captionz — llama.cpp backend: runs without Ollama.

Captionz downloads a prebuilt `llama-server` from the llama.cpp GitHub releases
(CUDA / Vulkan / CPU on Windows, Vulkan / CPU on Linux, Metal on macOS) and a
GGUF vision model (default: Qwen2.5-VL-3B-Instruct Q4_K_M + its mmproj) on first
use, starts the server on a free local port and talks to it through the
OpenAI-compatible /v1/chat/completions endpoint with base64 images.

Layout (settings.llamacpp_dir, default <app>/llamacpp):
    bin/<build>/llama-server(.exe) [+ dlls]      server binary (build = release tag)
    bin/current.json                               which build is active
    models/<name>/model.gguf                       weights
    models/<name>/mmproj.gguf                      vision projector
    models/<name>/model.json                       {name, source, added, ...}
    server.log                                     llama-server output

Models can be added from Hugging Face (any GGUF + mmproj) or imported from the
local Ollama store (~/.ollama/models): the manifest's "model" and "projector"
layers are hard-linked (no copy) into models/<name>/.

CLI: python captionz_models.py --help
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from captionz_core import APP_DIR, Backend, OllamaClient

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

DEFAULT_DIR = APP_DIR / "llamacpp"
GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"

# Default model: small, good, supported by llama.cpp's multimodal stack.
DEFAULT_MODEL = {
    "name": "Qwen2.5-VL-3B-Instruct-Q4_K_M",
    "repo": "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
    "model_file": "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    "mmproj_file": "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
}
# Other known-good HF GGUF vision models (name -> spec), offered in the UIs.
KNOWN_MODELS = {
    DEFAULT_MODEL["name"]: DEFAULT_MODEL,
    "Qwen2.5-VL-7B-Instruct-Q4_K_M": {
        "repo": "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
        "model_file": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
    },
    "gemma-3-4b-it-Q4_K_M": {
        "repo": "ggml-org/gemma-3-4b-it-GGUF",
        "model_file": "gemma-3-4b-it-Q4_K_M.gguf",
        "mmproj_file": "mmproj-model-f16.gguf",
    },
}

ProgressFn = "callable[[str], None] | None"


def _log(cb, msg: str) -> None:
    if cb:
        cb(msg)


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, progress=None, label: str = "") -> Path:
    """Stream a URL to dest (via dest.part), reporting MB and % when known."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Captionz"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, time.time()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 1.5:
                pct = f" {done * 100 // total}%" if total else ""
                _log(progress, f"↓ {label or dest.name}: {done >> 20} / {total >> 20} Mo{pct}")
                last = time.time()
    part.replace(dest)
    _log(progress, f"✔ {label or dest.name} ({dest.stat().st_size >> 20} Mo)")
    return dest


def _json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "Captionz", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# llama-server binary
# --------------------------------------------------------------------------- #
def _nvidia_cuda_version() -> float | None:
    """CUDA version supported by the NVIDIA driver (from nvidia-smi), or None."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
        # "CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.4" on newer ones
        m = re.search(r"CUDA(?: UMD)? Version:\s*([0-9]+\.[0-9]+)", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def pick_build(override: str = "") -> str:
    """Release asset flavour for this machine, e.g. 'win-cuda-13.3-x64'.
    `override` (settings.llamacpp_build) forces one, e.g. 'win-vulkan-x64'."""
    if override:
        return override
    sysname, arch = platform.system(), platform.machine().lower()
    x64 = arch in ("amd64", "x86_64")
    if sysname == "Windows":
        if not x64:
            return "win-cpu-arm64"
        cuda = _nvidia_cuda_version()
        if cuda:
            return "win-cuda-13.3-x64" if cuda >= 13.0 else "win-cuda-12.4-x64"
        return "win-vulkan-x64"
    if sysname == "Darwin":
        return "macos-arm64" if arch in ("arm64", "aarch64") else "macos-x64"
    if x64:
        return "ubuntu-vulkan-x64" if _nvidia_cuda_version() or _has_vulkan() else "ubuntu-x64"
    return "ubuntu-arm64"


def _has_vulkan() -> bool:
    return shutil.which("vulkaninfo") is not None


def latest_release() -> dict:
    """Most recent llama.cpp release that ships binaries (tags look like b12345)."""
    for rel in _json(GITHUB_RELEASES):
        if any(a["name"].startswith("llama-") and "-bin-" in a["name"] for a in rel.get("assets", [])):
            return rel
    raise RuntimeError("no llama.cpp release with binaries found")


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


def _find_server(root: Path) -> Path | None:
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    for p in root.rglob(exe):
        return p
    return None


class ServerBinary:
    def __init__(self, base_dir: Path, build_override: str = ""):
        self.bin_dir = base_dir / "bin"
        self.build_override = build_override

    @property
    def current_file(self) -> Path:
        return self.bin_dir / "current.json"

    def current(self) -> dict | None:
        try:
            info = json.loads(self.current_file.read_text("utf-8"))
            if Path(info["server"]).exists():
                return info
        except Exception:
            pass
        return None

    def path(self) -> Path | None:
        override = os.environ.get("CAPTIONZ_LLAMA_SERVER")
        if override and Path(override).exists():
            return Path(override)
        info = self.current()
        return Path(info["server"]) if info else None

    def ensure(self, progress=None) -> Path:
        p = self.path()
        if p:
            return p
        return self.install(progress)

    def install(self, progress=None, tag: str | None = None) -> Path:
        """Download and unpack the release for this machine (or `tag`)."""
        build = pick_build(self.build_override)
        rel = latest_release() if tag is None else next(
            r for r in _json(GITHUB_RELEASES) if r["tag_name"] == tag)
        tag = rel["tag_name"]
        assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
        main = next((n for n in assets if n.startswith(f"llama-{tag}-bin-{build}.")), None)
        if not main:
            raise RuntimeError(f"no llama.cpp asset for build '{build}' in release {tag}")
        target = self.bin_dir / f"{tag}-{build}"
        tmp = self.bin_dir / "downloads"
        _log(progress, f"llama.cpp {tag} ({build})")
        archive = download(assets[main], tmp / main, progress, main)
        _extract(archive, target)
        if "cuda" in build:  # CUDA runtime dlls (Windows) ship in a separate zip
            cudart = next((n for n in assets if n.startswith("cudart-") and build.split("win-")[-1] in n), None)
            if cudart:
                _extract(download(assets[cudart], tmp / cudart, progress, cudart), target)
        server = _find_server(target)
        if not server:
            raise RuntimeError(f"llama-server not found in {target}")
        if os.name != "nt":
            server.chmod(0o755)
        self.current_file.write_text(json.dumps({"tag": tag, "build": build, "server": str(server)}, indent=2), "utf-8")
        shutil.rmtree(tmp, ignore_errors=True)
        _log(progress, f"✔ llama-server prêt : {server}")
        return server

    def update(self, progress=None) -> str:
        """Install the newest release if it differs from the current one."""
        cur = self.current()
        rel = latest_release()
        if cur and cur.get("tag") == rel["tag_name"] and cur.get("build") == pick_build(self.build_override):
            _log(progress, f"llama-server déjà à jour ({cur['tag']})")
            return cur["tag"]
        old = cur and Path(cur["server"]).parent
        self.install(progress, rel["tag_name"])
        if old and old.exists():
            shutil.rmtree(old, ignore_errors=True)
        return rel["tag_name"]


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
class ModelRegistry:
    def __init__(self, base_dir: Path):
        self.models_dir = base_dir / "models"

    def list(self) -> list[str]:
        if not self.models_dir.exists():
            return []
        return sorted(p.name for p in self.models_dir.iterdir()
                      if (p / "model.gguf").exists() and (p / "mmproj.gguf").exists())

    def info(self, name: str) -> dict:
        try:
            return json.loads((self.models_dir / name / "model.json").read_text("utf-8"))
        except Exception:
            return {"name": name}

    def paths(self, name: str) -> tuple[Path, Path]:
        d = self.models_dir / name
        return d / "model.gguf", d / "mmproj.gguf"

    def has(self, name: str) -> bool:
        return name in self.list()

    def remove(self, name: str) -> None:
        shutil.rmtree(self.models_dir / name, ignore_errors=True)

    # ---- Hugging Face ------------------------------------------------------ #
    def add_from_hf(self, name: str, repo: str, model_file: str, mmproj_file: str, progress=None) -> str:
        d = self.models_dir / name
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "model.gguf").exists():
            download(HF_RESOLVE.format(repo=repo, file=model_file), d / "model.gguf", progress, model_file)
        if not (d / "mmproj.gguf").exists():
            download(HF_RESOLVE.format(repo=repo, file=mmproj_file), d / "mmproj.gguf", progress, mmproj_file)
        (d / "model.json").write_text(json.dumps({
            "name": name, "source": "huggingface", "repo": repo, "model_file": model_file,
            "mmproj_file": mmproj_file, "added": time.strftime("%Y-%m-%d %H:%M")}, indent=2), "utf-8")
        return name

    def add_known(self, name: str = DEFAULT_MODEL["name"], progress=None) -> str:
        spec = KNOWN_MODELS[name]
        return self.add_from_hf(name, spec["repo"], spec["model_file"], spec["mmproj_file"], progress)

    def ensure_default(self, progress=None) -> str:
        if not self.has(DEFAULT_MODEL["name"]):
            _log(progress, f"Téléchargement du modèle par défaut {DEFAULT_MODEL['name']} (≈ 2,9 Go)…")
            self.add_known(DEFAULT_MODEL["name"], progress)
        return DEFAULT_MODEL["name"]

    def update_from_source(self, name: str, progress=None) -> bool:
        """Re-download a Hugging Face model when the remote file size changed."""
        info = self.info(name)
        if info.get("source") != "huggingface":
            _log(progress, f"{name} : pas une source Hugging Face, rien à mettre à jour")
            return False
        changed = False
        for key, local in (("model_file", "model.gguf"), ("mmproj_file", "mmproj.gguf")):
            url = HF_RESOLVE.format(repo=info["repo"], file=info[key])
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Captionz"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    remote = int(r.headers.get("Content-Length") or 0)
            except Exception as e:  # noqa: BLE001
                _log(progress, f"{info[key]} : HEAD impossible ({e})")
                continue
            lp = self.models_dir / name / local
            if remote and lp.exists() and lp.stat().st_size != remote:
                _log(progress, f"{info[key]} a changé sur le Hub, nouveau téléchargement")
                lp.unlink()
                download(url, lp, progress, info[key])
                changed = True
        _log(progress, f"{name} : {'mis à jour' if changed else 'déjà à jour'}")
        return changed

    # ---- Ollama import ----------------------------------------------------- #
    @staticmethod
    def ollama_models_dir() -> Path:
        return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")

    @classmethod
    def ollama_manifest(cls, ollama_name: str) -> tuple[Path, dict]:
        """Manifest path + content for 'ns/name:tag' / 'name:tag' / 'name'."""
        name, _, tag = ollama_name.partition(":")
        tag = tag or "latest"
        ns, _, short = name.rpartition("/")
        ns = ns or "library"
        root = cls.ollama_models_dir() / "manifests"
        candidates = list(root.glob(f"*/{ns}/{short}/{tag}"))
        if not candidates:
            raise FileNotFoundError(f"modèle Ollama introuvable : {ollama_name} (dans {root})")
        return candidates[0], json.loads(candidates[0].read_text("utf-8"))

    @classmethod
    def ollama_vision_models(cls) -> list[str]:
        """Ollama models that have a vision projector layer (importable)."""
        out = []
        root = cls.ollama_models_dir() / "manifests"
        for mf in root.glob("*/*/*/*"):
            try:
                m = json.loads(mf.read_text("utf-8"))
            except Exception:
                continue
            types = {l.get("mediaType") for l in m.get("layers", [])}
            if "application/vnd.ollama.image.projector" in types:
                ns, name, tag = mf.parts[-3], mf.parts[-2], mf.name
                out.append((f"{name}:{tag}" if ns == "library" else f"{ns}/{name}:{tag}"))
        return sorted(out, key=str.lower)

    def import_from_ollama(self, ollama_name: str, name: str | None = None, progress=None) -> str:
        """Hard-link (or copy) the GGUF + projector blobs of an Ollama model."""
        _, manifest = self.ollama_manifest(ollama_name)
        layers = {l["mediaType"]: l for l in manifest.get("layers", [])}
        model = layers.get("application/vnd.ollama.image.model")
        proj = layers.get("application/vnd.ollama.image.projector")
        if not model:
            raise RuntimeError(f"{ollama_name} : pas de couche modèle dans le manifeste")
        if not proj:
            raise RuntimeError(f"{ollama_name} : pas de projecteur vision (modèle texte seul ?)")
        blobs = self.ollama_models_dir() / "blobs"
        name = name or re.sub(r"[^A-Za-z0-9._-]+", "_", ollama_name.split("/")[-1])
        d = self.models_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for layer, target in ((model, "model.gguf"), (proj, "mmproj.gguf")):
            src = blobs / layer["digest"].replace(":", "-")
            dst = d / target
            if dst.exists():
                continue
            if not src.exists():
                raise FileNotFoundError(f"blob manquant : {src}")
            # hard link (same drive) > symlink (other drive, needs Windows dev mode or admin) > copy
            linked = False
            for kind, fn in (("lien", os.link), ("lien symbolique", os.symlink)):
                try:
                    fn(src, dst)
                    _log(progress, f"{kind} {target} ← {src.name[:19]}… ({layer['size'] >> 20} Mo, sans copie)")
                    linked = True
                    break
                except OSError:
                    continue
            if not linked:
                _log(progress, f"copie {target} ({layer['size'] >> 20} Mo)…")
                shutil.copy2(src, dst)
        # Ollama also stores a system prompt and sampling params: keep them so the
        # model behaves like it does under Ollama.
        meta: dict = {"name": name, "source": "ollama", "ollama_name": ollama_name,
                      "added": time.strftime("%Y-%m-%d %H:%M")}
        sysl = layers.get("application/vnd.ollama.image.system")
        if sysl:
            try:
                meta["system"] = (blobs / sysl["digest"].replace(":", "-")).read_text("utf-8").strip()
            except Exception:
                pass
        parl = layers.get("application/vnd.ollama.image.params")
        if parl:
            try:
                params = json.loads((blobs / parl["digest"].replace(":", "-")).read_text("utf-8"))
                keep = {k: v for k, v in params.items()
                        if k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "stop")}
                if keep:
                    meta["params"] = keep
            except Exception:
                pass
        (d / "model.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        _log(progress, f"✔ {ollama_name} importé sous le nom {name}")
        return name


# --------------------------------------------------------------------------- #
# Server process
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServer:
    def __init__(self, server_bin: Path, log_file: Path, ctx: int = 8192, gpu_layers: int = 99):
        self.server_bin, self.log_file, self.ctx, self.gpu_layers = server_bin, log_file, ctx, gpu_layers
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self.no_think: bool | None = None
        self.port: int | None = None
        atexit.register(self.stop)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, model_name: str, model: Path, mmproj: Path, progress=None, timeout: int = 300,
              no_think: bool = True) -> None:
        if self.running() and self.model == model_name and self.no_think == no_think:
            return
        self.stop()
        self.port = _free_port()
        cmd = [str(self.server_bin), "-m", str(model), "--mmproj", str(mmproj), "--host", "127.0.0.1",
               "--port", str(self.port), "-c", str(self.ctx), "-ngl", str(self.gpu_layers), "--no-webui"]
        if no_think:
            # thinking off for templates that support it, zero budget otherwise (same idea as Ollama's think=false)
            cmd += ["--reasoning", "off", "--reasoning-budget", "0"]
        self.no_think = no_think
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_file, "ab")
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n".encode())
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=self.server_bin.parent,
                                     creationflags=flags)
        self.model = model_name
        _log(progress, f"llama-server : chargement de {model_name} sur le port {self.port}…")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                tail = self.log_file.read_bytes()[-1500:].decode("utf-8", "replace")
                raise RuntimeError(f"llama-server s'est arrêté (code {self.proc.returncode}) :\n{tail}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if r.status == 200:
                        _log(progress, f"✔ {model_name} chargé en {time.time() - t0:.0f} s")
                        return
            except Exception:
                pass
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(f"llama-server n'a pas répondu en {timeout} s (voir {self.log_file})")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except Exception:
                self.proc.kill()
        self.proc, self.model = None, None

    def chat(self, prompt: str, image_b64: str, temperature: float, max_tokens: int,
             system: str = "", params: dict | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]})
        payload: dict = {"messages": messages, "temperature": temperature}
        for k, v in (params or {}).items():          # Ollama-style sampling params
            if k == "stop":
                # thinking closers as stop words would cut the answer before it starts
                v = [x for x in v if not re.search(r"</?(think|thought)", x)]
                if v:
                    payload["stop"] = v
            elif k != "temperature":
                payload[k] = v
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode("utf-8"))
        choice = (data.get("choices") or [{}])[0]
        text = OllamaClient.strip_thinking((choice.get("message") or {}).get("content") or "")
        if choice.get("finish_reason") == "length" and not text:
            raise RuntimeError(f"limite de {max_tokens} tokens atteinte sans caption ; augmente « tokens max »")
        return text


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class LlamaCppBackend(Backend):
    name = "llamacpp"
    _server: LlamaServer | None = None     # one server per process, shared by backend instances
    _lock = threading.Lock()

    def __init__(self, base_dir: Path | str | None = None, model: str = "", build_override: str = "",
                 max_tokens: int = 1024, ctx: int = 8192, no_think: bool = True):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_DIR
        self.registry = ModelRegistry(self.base_dir)
        self.binary = ServerBinary(self.base_dir, build_override)
        self.model = model
        self.max_tokens = max_tokens
        self.ctx = ctx
        self.no_think = no_think
        self.on_progress = None   # callable(str) set by the UIs for download / load messages

    # ---- interface ----------------------------------------------------------- #
    def list_models(self) -> list[str]:
        models = self.registry.list()
        return models or [DEFAULT_MODEL["name"]]   # the default is offered even before download

    def is_loaded(self, model: str) -> bool:
        srv = LlamaCppBackend._server
        return bool(srv and srv.running() and srv.model == (model or self.model) and srv.no_think == self.no_think)

    def load(self, model: str) -> None:
        name = model or self.model or DEFAULT_MODEL["name"]
        with LlamaCppBackend._lock:
            server_bin = self.binary.ensure(self.on_progress)
            if not self.registry.has(name):
                if name in KNOWN_MODELS:
                    _log(self.on_progress, f"Téléchargement de {name}…")
                    self.registry.add_known(name, self.on_progress)
                else:
                    raise FileNotFoundError(f"modèle llama.cpp inconnu : {name} (voir captionz_models.py)")
            if LlamaCppBackend._server is None or LlamaCppBackend._server.server_bin != server_bin:
                if LlamaCppBackend._server:
                    LlamaCppBackend._server.stop()
                LlamaCppBackend._server = LlamaServer(server_bin, self.base_dir / "server.log", ctx=self.ctx)
            m, p = self.registry.paths(name)
            LlamaCppBackend._server.start(name, m, p, self.on_progress, no_think=self.no_think)

    def caption(self, model, prompt, image_path, *, temperature=0.2, max_side=1024) -> str:
        name = model or self.model or DEFAULT_MODEL["name"]
        if not self.is_loaded(name):
            self.load(name)
        b64 = OllamaClient.encode_image(Path(image_path), max_side)
        info = self.registry.info(name)
        return LlamaCppBackend._server.chat(prompt, b64, float(temperature), self.max_tokens,
                                            info.get("system", ""), info.get("params"))

    @classmethod
    def shutdown(cls) -> None:
        if cls._server:
            cls._server.stop()
