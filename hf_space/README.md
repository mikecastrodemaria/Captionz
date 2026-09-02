---
title: Captionz
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
---

# Captionz

Batch image captioning with vision-language models. Upload images (or a folder), compose the prompt
(caption type × length × options × character name), run, edit captions, download the zip.

Backend on Spaces: `transformers` (Qwen2.5-VL by default) with ZeroGPU. Set `CAPTIONZ_HF_MODEL` in the
Space secrets/variables to change the default model.

Source: https://github.com/mikecastrodemaria/Captionz
