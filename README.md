# Captionz

Batch image captioning with a **vision** model served by [Ollama](https://ollama.com). One core, four front ends: a command line, a Tkinter desktop app (standard library only), a NiceGUI web UI (`--ui web`), and a Gradio app made for Hugging Face Spaces.

The Ollama layer reuses proven patterns from crispz-studio (`cz_ollama.py`): vision detection through `/api/show` with a name-based fallback, JPEG downscaling before upload, stripping of `<think>` blocks from "thinking" models, `keep_alive` / CPU mode so the model does not hog VRAM.

## Features

- Connect to any Ollama server (configurable URL, local or remote)
- Automatic list of installed models **with the vision capability**
- Sources: a single file, a selection of files, a folder (recursive or not), or **paste from the clipboard** (Ctrl+V: a screenshot or copied image is saved as PNG in `pasted/`, copied files are added directly)
- **Composed prompt**: caption type (formal / casual descriptive, training caption, Stable Diffusion prompt, Booru tags, art critic, product listing, social media post, short sentence) × length (very short → very long) × checkable options (lighting, camera angle, watermark, JPEG artifacts, camera settings, keep it PG, aesthetic quality, composition, depth of field, sfw/nsfw, avoid words blocked by Midjourney and similar filters…)
- Character name: `{name}` is substituted in the prompt and the options
- Custom prompt that overrides everything, and a **preview of the final prompt** actually sent
- **Image preview**, **editable caption** with a Save button
- Existing captions: skip, overwrite, or append
- Caption everything, only the selection, or the displayed image
- Prefix / suffix (trigger word), output extension, single-line output
- `keep_alive` (0 = unload the model after each image), max image side in px, CPU mode
- Model blocklist through `settings.json` (`vision_blocklist`)
- **Dark mode**, background processing, progress bar, log, clean stop
- Settings are remembered in `settings.json`

Captions are written next to each image: `image.jpg` → `image.txt`. The UI language is French.

## Requirements

- Python 3.10+ (Tkinter ships with the official Windows installer)
- Optional: `pip install pillow` for image preview and downscaling before upload (faster, fewer tokens)
- Ollama running with at least one vision model, for example:

```bash
ollama pull qwen3-vl:8b
```

## Install and run

The scripts create a `.venv` virtual environment and install Pillow.

| Platform | Install | Desktop UI | Web UI |
|---|---|---|---|
| Windows (cmd) | `install.bat` | `start.bat` | `startweb.bat` |
| Windows (PowerShell) | `.\install.ps1` | `.\start.ps1` | `.\startweb.ps1` |
| Linux / macOS | `./install.sh` | `./start.sh` | `./startweb.sh` |

If PowerShell refuses to run the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Without the scripts:

```bash
python app.py                                  # Tkinter desktop UI (default)
python app.py --ui web                         # NiceGUI web UI, opens http://127.0.0.1:8080
python app.py --ui web --port 8090 --host 0.0.0.0 --no-browser   # expose on the LAN, no auto-open
```

The web UI (`webui.py`) needs `pip install nicegui` (done by the install scripts). Both UIs share `captionz_core.py` (Ollama client, prompt composition, settings, background captioning) and the same `settings.json`.

Web UI sources: a local path (file or folder) typed in the page, browser upload (files are copied to `pasted/uploads`), or Ctrl+V of a screenshot / copied image anywhere in the page.

## Architecture

| File | Role |
|---|---|
| `captionz_core.py` | Everything UI-independent: Ollama client, prompt composition, settings, caption policy (skip / overwrite / append, prefix, suffix), `run_jobs()` batch generator, backend abstraction |
| `captionz_hf.py` | `transformers` backend (Qwen2.5-VL and friends), ZeroGPU-aware, used on Spaces |
| `cli.py` | Command line on top of the core |
| `app.py` | Tkinter desktop UI + entry point (`--ui tk|web`) |
| `webui.py` | NiceGUI web UI |
| `gradio_app.py` | Gradio UI (Spaces or local) |
| `hf_space/`, `deploy_space.py` | Space entry files and an upload script |
| `bench.py` | Model benchmark |

All front ends read and write the same `settings.json`.

## Command line

```bash
python cli.py photo.jpg
python cli.py ./dataset --recursive --type "Booru tag list" --length short
python cli.py ./dataset --model qwen3-vl:8b --option 3 --option 15 --name Lea --prefix "lea_style, "
python cli.py ./dataset --existing overwrite --json report.json
python cli.py ./dataset --backend hf --hf-model Qwen/Qwen2.5-VL-3B-Instruct
python cli.py --list-models
python cli.py --list-options        # option numbers for --option
python cli.py --show-prompt --type "Stable Diffusion prompt" --option 18
python cli.py --type "Short sentence" --save     # persist flags into settings.json
```

Defaults come from `settings.json`; flags override them for the run. Exit code is 1 when at least one image failed.

## Gradio and Hugging Face Spaces

```bash
python gradio_app.py                      # local, backend Ollama, http://127.0.0.1:7860
python gradio_app.py --backend hf         # local with transformers (needs torch + transformers)
python gradio_app.py --share              # temporary public link
```

Deploy to a Space (Gradio SDK, ZeroGPU hardware by default):

```bash
pip install huggingface_hub
huggingface-cli login
python deploy_space.py <user>/captionz --create
```

`deploy_space.py` uploads only what the Space needs (`captionz_core.py`, `captionz_hf.py`, `gradio_app.py`, and the entry files from `hf_space/`). On Spaces the backend is `transformers` with Qwen2.5-VL-3B by default; set the `CAPTIONZ_HF_MODEL` variable in the Space settings to change it. Sources on Spaces are uploads and clipboard paste; captions are edited in place and downloaded as a zip.

## Model benchmark

```bash
python bench.py "<image or folder>" [--exclude name1,name2] [--models a,b] [--prompt "…"] [--max-side 1024]
```

Loops model → images (each model is loaded once), writes one `<image>__<model>.txt` per caption, a `README.md` with a timing table and every caption, and `results.json`. Past results live in `benchmarks/`.

Results on an RTX 5090, 24 images, "paragraph" prompt, images downscaled to 1024 px:

| Model | Avg. per image | Manual check (3 images) |
|---|---:|:---:|
| Agents-A1 4B Kimi | 4.2 s | 3/3 |
| qwen3.6 | 5.4 s | 2/3 |
| qwenpaw 9B | 5.8 s | 2/3 |
| Qwable 9B Q6 | 6.7 s | 3/3 |
| muse-glimmer 28B | 7.1 s | 3/3 |
| gemma-4 31B vision | 11.6 s | 3/3 |

## Supported image formats

jpg, jpeg, png, webp, bmp, gif, tif, tiff.

## License

MIT
