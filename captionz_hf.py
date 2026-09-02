"""
Captionz — Hugging Face `transformers` backend (for Spaces or any GPU box).

Loads a vision-language model with AutoModelForImageTextToText + AutoProcessor
(Qwen2.5-VL, Qwen2-VL, LLaVA-OneVision, SmolVLM, Gemma 3, …) and captions with
the same prompt as the Ollama backend.

ZeroGPU (Hugging Face Spaces): the decorated `_generate` runs in a forked
process, so the model is loaded on CPU in the parent process (`_load`, module
level state) and only moved to CUDA inside the GPU function. Locally with a GPU
the same code just runs in-process.

Dependencies (not needed for the local Ollama UIs):
    pip install torch transformers accelerate pillow
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from captionz_core import Backend, OllamaClient

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

# Curated list shown in the UI; any hub id that AutoModelForImageTextToText loads works.
HF_MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceTB/SmolVLM-Instruct",
    "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "google/gemma-3-4b-it",
]
DEFAULT_HF_MODEL = os.environ.get("CAPTIONZ_HF_MODEL", HF_MODELS[0])
MAX_NEW_TOKENS = int(os.environ.get("CAPTIONZ_MAX_NEW_TOKENS", "512"))

try:  # ZeroGPU on Spaces: decorate the GPU-bound function
    import spaces  # type: ignore
    _gpu = spaces.GPU(duration=120)
except Exception:  # noqa: BLE001
    def _gpu(fn):  # type: ignore
        return fn

# ---- module-level model state (shared with the forked GPU worker) ---------- #
_STATE: dict = {"id": None, "model": None, "processor": None}
_LOCK = threading.Lock()


def _load(model_id: str) -> None:
    """Load processor + model on CPU (idempotent). Safe to call outside the GPU."""
    with _LOCK:
        if _STATE["id"] == model_id:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        model.eval()
        _STATE.update(id=model_id, model=model, processor=processor)


@_gpu
def _generate(model_id: str, prompt: str, img, temperature: float, max_new_tokens: int) -> str:
    import torch
    _load(model_id)
    proc, model = _STATE["processor"], _STATE["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(model.device) != device:
        model.to(device)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    chat = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=[chat], images=[img], return_tensors="pt").to(device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return proc.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


class HFBackend(Backend):
    name = "hf"

    def __init__(self, model_id: str | None = None, max_new_tokens: int = MAX_NEW_TOKENS):
        self.model_id = model_id or DEFAULT_HF_MODEL
        self.max_new_tokens = max_new_tokens

    def list_models(self) -> list[str]:
        ids = list(HF_MODELS)
        if self.model_id not in ids:
            ids.insert(0, self.model_id)
        return ids

    def preload(self) -> None:
        """Load the default model on CPU (call at startup on Spaces)."""
        _load(self.model_id)

    def caption(self, model, prompt, image_path, *, temperature=0.2, max_side=1024) -> str:
        model_id = model or self.model_id
        if Image is None:
            raise RuntimeError("Pillow is required for the transformers backend")
        img = Image.open(Path(image_path)).convert("RGB")
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side))
        _load(model_id)  # in the parent process, so the forked GPU worker inherits it
        return OllamaClient.strip_thinking(_generate(model_id, prompt, img, float(temperature), self.max_new_tokens))
