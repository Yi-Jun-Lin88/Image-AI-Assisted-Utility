# Image AI Utility

**Live Demo:** [Open the Streamlit app](https://image-ai-assisted-utility-f87dyujzdggtbfwbyrvvdd.streamlit.app/)

Image AI Utility is a Streamlit demo for AI-Assisted PM interviews. It productizes the thesis direction **"Image Segmentation Based on Depth Estimation"** into a shareable single-image workflow that shows how depth can become a reusable product primitive for image editing and image understanding.

## What It Does

- Accepts either a built-in sample image or an uploaded PNG, JPG, JPEG, or WEBP image.
- Estimates a depth map with `depth-anything/Depth-Anything-V2-Small-hf`.
- Builds a depth-aware subject extraction using automatic depth thresholding plus Scharr gradient refinement.
- Generates a portrait bokeh output by preserving the foreground subject and blurring background depth layers.
- Produces a subject-focused caption through the caption provider boundary, with a deterministic fallback for the MVP.
- Runs image classification with a HuggingFace image-classification model and a graceful fallback.
- Provides downloads for generated depth map, extracted subject, and portrait bokeh images.

## Why HuggingFace Instead Of CoreML

The thesis implementation used `DepthAnythingV2SmallF16.mlpackage`, which is a strong fit for macOS and Apple Silicon. This demo uses HuggingFace `transformers` instead because the interview artifact needs to be easy to run and share on CPU-friendly deployment targets such as Streamlit Cloud or Hugging Face Spaces.

That tradeoff keeps the MVP portable: reviewers can open the app without a local Apple Silicon/CoreML setup, while the architecture still preserves a clear model boundary for revisiting the thesis CoreML package later.

## MVP Boundaries

In scope:

- Single-image upload or sample-image demo flow.
- Automatic depth estimation, foreground masking, bokeh generation, captioning, classification, and downloads.
- Graceful degradation when optional or remote model steps fail.
- Modular pipeline files that make later model swaps straightforward.

Out of scope:

- Batch processing, accounts, persistence, or project management features.
- Manual mask editing, brush tools, threshold sliders, or pixel-level review workflows.
- Production matting quality guarantees across all hair, glass, motion blur, and low-contrast cases.
- Mobile/CoreML packaging, GPU-specific optimization, and hosted production observability.

## Local Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The full app downloads HuggingFace models on first run. Initial startup can take longer depending on network speed and local cache state.

## Tests

```bash
pytest tests -v
```

The deterministic test suite avoids HuggingFace downloads by testing local logic and injected pipeline boundaries. The full Streamlit app may download depth and classification models the first time it runs.

## Deployment Path

The intended lightweight deployment path is Streamlit Cloud or Hugging Face Spaces:

1. Push the repository with `app.py`, `pipeline/`, `requirements.txt`, `tests/`, and this README.
2. Configure the platform to run `streamlit run app.py`.
3. Allow first-run HuggingFace model downloads and platform-level caching.
4. Add `VISION_API_KEY` only when a real caption provider replaces the MVP fallback.

This path favors interview shareability over device-specific acceleration. A later production path can reintroduce CoreML for Apple platforms or add stronger segmentation/matting services behind the existing pipeline interfaces.

## Interview Narrative

This project is not just a wrapper around a single background-removal model. The product bet is that depth is a reusable primitive: one depth estimate can support subject isolation, background replacement, portrait bokeh, editing suggestions, and downstream image understanding.

The MVP deliberately keeps segmentation automatic with Scharr gradient refinement so the demo remains clear in an interview setting. Its modular architecture also leaves room for stronger segmentation, matting, SAM-style prompting, or CoreML acceleration without rewriting the Streamlit product surface.
