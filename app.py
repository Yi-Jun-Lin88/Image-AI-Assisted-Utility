from __future__ import annotations

import os
from io import BytesIO
from hashlib import sha256
from pathlib import Path

import streamlit as st
from PIL import Image

from pipeline.image_io import (
    SUPPORTED_UPLOAD_TYPES,
    image_to_png_bytes,
    register_supported_image_openers,
)
from pipeline.orchestrator import run_image_pipeline
from pipeline.types import PipelineResult


st.set_page_config(
    page_title="Image AI Utility",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_IMAGE_PATH = Path(__file__).parent / "sample_data" / "sample.jpg"
PROCESSING_MODES = {
    "Fast preview": 512,
    "Refined output": 1024,
}
register_supported_image_openers()


def load_uploaded_image() -> Image.Image | None:
    uploaded_file = st.file_uploader(
        "Upload a PNG, JPG, JPEG, WEBP, HEIF, or HEIC image",
        type=SUPPORTED_UPLOAD_TYPES,
    )
    if uploaded_file is None:
        return None

    try:
        data = BytesIO(uploaded_file.getvalue())
        return Image.open(data).copy()
    except Exception as exc:
        st.error(f"Could not read image: {exc}")
        return None


def load_sample_image() -> Image.Image:
    return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")


def show_image_slot(title: str, image: Image.Image | None, *, caption: str | None = None) -> None:
    st.subheader(title)
    if image is None:
        st.info("Not available for this run.")
        return
    st.image(image, caption=caption, use_container_width=True)


def show_download(label: str, image: Image.Image | None, file_name: str) -> None:
    if image is None:
        return
    st.download_button(
        label,
        data=image_to_png_bytes(image),
        file_name=file_name,
        mime="image/png",
        use_container_width=True,
        on_click="ignore",
    )


def show_caption(result: PipelineResult) -> None:
    st.subheader("Subject Caption")
    st.write(result.caption.text)
    provider = result.caption.provider
    if result.caption.used_fallback:
        st.caption(f"Provider: {provider} fallback")
    else:
        st.caption(f"Provider: {provider}")


def show_classification(result: PipelineResult) -> None:
    st.subheader("Classification")
    rows = [
        {
            "Label": item.label,
            "Score": f"{item.score:.2%}",
            "Source": "fallback" if item.used_fallback else "model",
        }
        for item in result.classifications
    ]
    if rows:
        st.table(rows)
    else:
        st.info("No classifications returned.")


def show_processing_notes(errors: list[str]) -> None:
    if not errors:
        return
    with st.expander("Processing notes", expanded=True):
        for error in errors:
            st.error(error)


st.title("Image AI Utility")
st.caption("Depth estimation x background removal x portrait bokeh x image understanding.")

st.sidebar.header("Input")
image_source = st.sidebar.radio(
    "Image source",
    options=["Sample image", "Upload image"],
)
with st.sidebar.expander("Advanced", expanded=False):
    api_key = st.text_input(
        "Vision API key (future provider placeholder)",
        value=os.getenv("VISION_API_KEY", ""),
        type="password",
        help="The MVP uses deterministic fallback captioning. This field marks the future Vision LLM integration point.",
    )
    st.caption("Vision captioning is not enabled in this MVP. Captioning currently uses fallback text even when a key is entered.")
st.sidebar.info(
    "MVP boundary: this desktop demo uses HuggingFace transformers instead of the "
    "thesis CoreML package. It supports a single-image workflow with automatic "
    "depth-aware processing and no manual threshold controls."
)

if image_source == "Upload image":
    st.subheader("Upload Image")
    input_image = load_uploaded_image()
    if input_image is None:
        st.info("Choose an image file above to start.")
        st.stop()
else:
    input_image = load_sample_image()

preview_columns = st.columns([1, 2])
with preview_columns[0]:
    st.subheader("Input Preview")
    st.image(input_image, use_container_width=True)

control_columns = st.columns(2)
with control_columns[0]:
    subject_strength = st.slider(
        "Subject strength",
        min_value=35,
        max_value=80,
        value=60,
        help="Higher values keep a tighter foreground subject.",
    )
with control_columns[1]:
    bokeh_strength = st.slider(
        "Bokeh strength",
        min_value=2,
        max_value=24,
        value=12,
        help="Higher values create stronger background blur.",
    )
processing_mode = st.segmented_control(
    "Processing mode",
    options=list(PROCESSING_MODES.keys()),
    default="Fast preview",
    help="Fast preview favors speed. Refined output processes a larger image for cleaner results.",
)
if processing_mode is None:
    processing_mode = "Fast preview"
max_side = PROCESSING_MODES[processing_mode]

image_fingerprint = sha256(image_to_png_bytes(input_image)).hexdigest()
run_key = f"{image_source}:{image_fingerprint}:{subject_strength}:{bokeh_strength}:{processing_mode}"
last_key = st.session_state.get("last_run_key")
result: PipelineResult | None = st.session_state.get("last_result")

refresh_requested = st.button("Refresh output", use_container_width=True)
if refresh_requested or result is None or last_key != run_key:
    with st.spinner("Running image pipeline..."):
        result = run_image_pipeline(
            input_image,
            api_key=api_key or None,
            subject_strength=subject_strength,
            bokeh_strength=bokeh_strength,
            max_side=max_side,
        )
    st.session_state["last_result"] = result
    st.session_state["last_run_key"] = run_key
    last_key = run_key

if result is None:
    st.warning("Preparing outputs.")
    st.stop()

st.caption(f"Processed in {processing_mode} mode at max side {max_side}px.")

image_row_one = st.columns(2)
with image_row_one[0]:
    show_image_slot("Original", result.original)
with image_row_one[1]:
    show_image_slot("Depth Map", result.depth_map)
    show_download("Download depth map", result.depth_map, "depth-map.png")

image_row_two = st.columns(2)
with image_row_two[0]:
    show_image_slot("Depth-Aware Subject", result.subject)
    show_download("Download subject", result.subject, "depth-aware-subject.png")
with image_row_two[1]:
    show_image_slot("Portrait Bokeh", result.bokeh)
    show_download("Download portrait bokeh", result.bokeh, "portrait-bokeh.png")

show_caption(result)
show_classification(result)

show_processing_notes(result.errors)

st.header("PM Decision Notes")
st.markdown(
    """
- Depth is the product primitive: one estimate supports subject isolation, background removal, and bokeh.
- Scharr gradient keeps the MVP automatic by deriving subject edges from depth without manual thresholds.
- HuggingFace models make the desktop prototype shareable before revisiting the thesis CoreML package.
- Next product comparisons should evaluate SAM, matting quality, model swaps, and a future mobile path.
"""
)
