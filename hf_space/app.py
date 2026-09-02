"""Hugging Face Space entry point: `python app.py` (Gradio SDK)."""
from gradio_app import build

try:  # load the default model on CPU at startup so ZeroGPU workers inherit it
    from captionz_hf import HFBackend
    HFBackend().preload()
except Exception as e:  # noqa: BLE001
    print(f"model preload skipped: {e}")

demo = build("hf")

if __name__ == "__main__":
    demo.launch()
