# Image AI Utility Demo Design

Date: 2026-05-23

## Purpose

Image AI Utility is a desktop-first interactive demo for AI-Assisted PM interviews. It demonstrates how a research idea from monocular depth estimation can become a product-shaped image utility.

The demo centers on a single pipeline:

1. Upload or select one image.
2. Estimate depth with a cross-platform HuggingFace model.
3. Use depth and Scharr gradient information to extract the foreground subject.
4. Simulate portrait-mode bokeh with depth-layered Gaussian blur.
5. Generate a subject-focused image description with a Vision LLM.
6. Classify the image with a lightweight HuggingFace vision model.
7. Download result images.

The goal is not to claim production-grade segmentation quality in the MVP. The goal is to show product judgment: why depth information matters, how a thesis technique can become a usable workflow, and how technical constraints shape deployment choices.

## Audience

The demo should work for both technical and non-technical interviewers.

Technical interviewers should see:

- Monocular depth estimation as the core technical primitive.
- Scharr gradient use on the depth map for contour and detail refinement.
- Clear model and deployment trade-offs.
- Modular code that can later swap in stronger models.

Non-technical interviewers should see:

- A simple single-image workflow.
- A visual before-and-after transformation.
- A clear connection to familiar user needs, especially background removal and portrait-mode bokeh.
- A concise explanation of product decisions and MVP boundaries.

## Product Positioning

The demo is a productized extension of the master's thesis topic, "Image Segmentation Based on Depth Estimation."

The thesis used CoreML through `DepthAnythingV2SmallF16.mlpackage`, which works well for local macOS and Apple Silicon experiments but is not suitable for Streamlit Cloud or Hugging Face Spaces deployment. The demo should therefore use the HuggingFace `transformers` version of Depth Anything V2 Small:

```python
from transformers import pipeline

depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
)
```

This makes the demo easier to share as a link and aligns with the AI-Assisted PM story: the research prototype used a local optimized path, while the interview demo chooses a cross-platform path that favors accessibility and reliable sharing.

## MVP Scope

### In Scope

- Single-image upload.
- Built-in sample image mode.
- Depth estimation with `depth-anything/Depth-Anything-V2-Small-hf`.
- Background removal / subject extraction using depth map plus Scharr gradient.
- Portrait bokeh simulation using depth-layered Gaussian blur.
- Image description generation through a Vision LLM API when configured.
- Local fallback caption when no Vision API key is available.
- Image classification with a HuggingFace image classification model, preferably a ViT pipeline for the MVP.
- Result image downloads.
- Product notes explaining depth-first decisions, technical trade-offs, and MVP boundaries.

### Out of Scope

- Manual threshold controls.
- Batch processing.
- Full model comparison between Depth Anything, SAM, DeepLabV3+, and rembg.
- Mobile optimization.
- Account system, persistence, gallery management, or sharing features.
- Production-grade segmentation quality guarantees.

## Recommended Approach

Use a Streamlit app with a real but lightweight pipeline.

This combines:

- A usable web demo that interviewers can click through.
- A guaranteed sample mode for polished interview presentation.
- A real upload path that demonstrates engineering feasibility.
- Product narrative embedded directly in the interface.

This is preferred over a pure mockup because it better supports the engineering and research-extension story. It is preferred over a fully production-grade local model stack because the MVP should remain deployable, explainable, and stable in interview settings.

## User Flow

1. User opens the Streamlit app.
2. User chooses either sample image mode or uploads one image.
3. App resizes the image to a reasonable processing size.
4. App runs the depth estimation pipeline.
5. App computes a depth-normalized map and Scharr gradient map.
6. App estimates a foreground mask automatically.
7. App produces a subject extraction result.
8. App produces a bokeh result using depth-aware blur layers.
9. App generates or falls back to a subject-focused caption.
10. App runs image classification.
11. User reviews all outputs and can download result images.

## Interface Structure

The first screen should be the tool itself, not a marketing landing page.

### Sidebar

- Image source selector: sample image or upload.
- File uploader.
- Optional Vision API provider/key configuration.
- Short MVP boundary note:
  - Cross-platform HuggingFace pipeline.
  - CoreML remains part of the thesis context, not the deployed demo dependency.
  - Single-image desktop MVP.

### Main Area

Top result row:

- Original image.
- Depth map.
- Subject extraction.

Second result row:

- Bokeh result.
- Generated caption.
- Image classification.

Bottom section:

- PM decision notes.
- Explanation of why depth-first extraction is meaningful.
- Explanation of why Scharr gradient is used.
- Future improvement path: SAM hybrid refinement, better matting, model comparison, and mobile optimization.

## Technical Architecture

The app should be implemented as Python plus Streamlit. Core behavior should be split into small modules rather than concentrated in `app.py`.

Suggested structure:

```text
app.py
pipeline/
  __init__.py
  bokeh.py
  caption.py
  classify.py
  depth.py
  mask.py
  types.py
sample_data/
  sample.jpg
README.md
requirements.txt
```

### Module Responsibilities

`app.py`

- Owns Streamlit UI.
- Handles upload/sample selection.
- Displays progress states.
- Renders images, captions, classifications, PM notes, and download buttons.

`pipeline/depth.py`

- Loads and caches the HuggingFace depth estimation pipeline.
- Produces a normalized depth map and a displayable depth visualization.

`pipeline/mask.py`

- Normalizes depth values.
- Applies Scharr gradient to the depth map.
- Builds an automatic foreground mask.
- Produces subject extraction output.

`pipeline/bokeh.py`

- Applies depth-layered Gaussian blur.
- Keeps likely foreground regions sharper.
- Produces the portrait-mode result.

`pipeline/caption.py`

- Calls a Vision LLM when credentials are configured.
- Provides a local fallback caption when no key is present or the API call fails.

`pipeline/classify.py`

- Runs a HuggingFace image classification pipeline.
- Returns top labels and confidence values.

`pipeline/types.py`

- Defines shared dataclasses or typed structures for pipeline results.
- Keeps UI and model implementation details loosely coupled.

## Model Strategy

Depth estimation:

- Use `depth-anything/Depth-Anything-V2-Small-hf` through `transformers.pipeline("depth-estimation", ...)`.
- Cache model loading with Streamlit cache primitives.
- Prefer CPU compatibility for deployment.

Classification:

- Use a HuggingFace image-classification pipeline for the MVP.
- Prefer ViT for a one-line inference path.
- Treat CLIP as a later extension, because open-vocabulary classification requires candidate label design.

Captioning:

- Implement a provider boundary that can support OpenAI or Claude Vision later.
- MVP can read an API key from Streamlit secrets or environment variables.
- If no key is available, continue with a deterministic fallback caption.

## Error Handling And Degradation

The demo should fail softly.

- If model download or model loading fails, show a clear error and keep the sample/product explanation visible.
- If Vision API credentials are missing, use fallback captioning.
- If the uploaded image is too large, resize before inference.
- If processing is slow on CPU, show clear progress states.
- If mask quality is poor, do not hide the limitation. PM notes should explain this as a depth-first MVP and identify SAM hybrid refinement as a future improvement.
- If any downstream step fails, still display successful upstream outputs when possible.

## Testing And Verification

MVP verification should focus on demo reliability and pipeline completeness.

Required checks:

- App starts locally with Streamlit.
- Sample image runs through the full pipeline.
- Uploaded image runs through the full pipeline.
- Depth map, subject extraction, bokeh result, caption, and classification all render.
- Vision API key absence does not block the demo.
- Download buttons produce valid image files.
- README explains local setup, deployment path, MVP boundaries, and interview narrative.

Model quality should be assessed visually, but the first implementation does not need benchmark metrics.

## Future Extensions

- Add SAM or DeepLabV3+ hybrid refinement.
- Add side-by-side model comparison as an advanced mode.
- Add manual controls for threshold, blur strength, and focus depth.
- Add batch processing.
- Add mobile layout optimization.
- Add stronger matting for hair and fine object boundaries.
- Add CLIP-based candidate label design for product-specific classification.
- Add a polished case-study page if the demo later becomes part of a portfolio site.

## Interview Narrative

The recommended explanation:

"My thesis explored image segmentation based on monocular depth estimation. For this demo, I productized that research into an AI image utility. Instead of directly wrapping SAM or rembg, I started from depth because depth gives a product-relevant signal: it can support both foreground extraction and portrait-mode bokeh. I also moved from a CoreML local model to a HuggingFace transformers model so the demo can be deployed and shared. The MVP intentionally keeps the workflow single-image and automatic, because the goal is to make the value understandable in an interview setting while leaving clear extension points for stronger segmentation and model comparison."
