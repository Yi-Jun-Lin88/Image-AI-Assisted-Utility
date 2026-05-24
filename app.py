from __future__ import annotations

import os
from io import BytesIO
from hashlib import sha256

import streamlit as st
from PIL import Image, ImageDraw

from pipeline.image_io import image_to_png_bytes
from pipeline.orchestrator import run_image_pipeline
from pipeline.types import PipelineResult


st.set_page_config(page_title="Image AI Utility", layout="wide")


def load_uploaded_image() -> Image.Image | None:
    uploaded_file = st.sidebar.file_uploader(
        "Image file",
        type=["png", "jpg", "jpeg", "webp"],
    )
    if uploaded_file is None:
        return None

    try:
        data = BytesIO(uploaded_file.getvalue())
        return Image.open(data).copy()
    except Exception as exc:
        st.sidebar.error(f"Could not read image: {exc}")
        return None


def load_sample_image() -> Image.Image:
    image = Image.new("RGB", (900, 600), color=(168, 202, 224))
    draw = ImageDraw.Draw(image)

    for y in range(600):
        if y < 330:
            blue = 224 - int(y * 0.18)
            draw.line([(0, y), (900, y)], fill=(168, 202, blue))
        else:
            green = 138 - int((y - 330) * 0.08)
            draw.line([(0, y), (900, y)], fill=(74, max(96, green), 88))

    draw.ellipse((90, 72, 210, 192), fill=(242, 206, 96))
    draw.rectangle((0, 330, 900, 600), fill=(78, 126, 92))
    draw.ellipse((280, 190, 620, 560), fill=(68, 82, 106))
    draw.ellipse((340, 100, 560, 320), fill=(225, 190, 154))
    draw.rectangle((392, 320, 508, 530), fill=(188, 92, 86))
    draw.polygon([(392, 348), (260, 470), (330, 505), (425, 392)], fill=(188, 92, 86))
    draw.polygon([(508, 348), (640, 470), (570, 505), (475, 392)], fill=(188, 92, 86))
    draw.ellipse((392, 162, 430, 200), fill=(42, 44, 52))
    draw.ellipse((470, 162, 508, 200), fill=(42, 44, 52))
    draw.arc((390, 178, 510, 258), start=20, end=160, fill=(90, 52, 50), width=6)

    return image


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
        {"Label": item.label, "Score": f"{item.score:.2%}"}
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
api_key = st.sidebar.text_input(
    "Vision API key",
    value=os.getenv("VISION_API_KEY", ""),
    type="password",
)
st.sidebar.info(
    "MVP boundary: this desktop demo uses HuggingFace transformers instead of the "
    "thesis CoreML package. It supports a single-image workflow with automatic "
    "depth-aware processing and no manual threshold controls."
)

if image_source == "Upload image":
    input_image = load_uploaded_image()
    if input_image is None:
        st.warning("Upload a PNG, JPG, JPEG, or WEBP image to start.")
        st.stop()
else:
    input_image = load_sample_image()

preview_columns = st.columns([1, 2])
with preview_columns[0]:
    st.subheader("Input Preview")
    st.image(input_image, use_container_width=True)

image_fingerprint = sha256(image_to_png_bytes(input_image)).hexdigest()
run_key = f"{image_source}:{image_fingerprint}:{bool(api_key)}"
last_key = st.session_state.get("last_run_key")
result: PipelineResult | None = st.session_state.get("last_result")

run_requested = st.button("Run pipeline", type="primary", use_container_width=True)
if run_requested:
    with st.spinner("Running image pipeline..."):
        result = run_image_pipeline(input_image, api_key=api_key or None)
    st.session_state["last_result"] = result
    st.session_state["last_run_key"] = run_key
    last_key = run_key

if result is None:
    st.warning("Run the pipeline to generate outputs.")
    st.stop()

if last_key != run_key:
    st.info("Input settings changed. Click Run pipeline to refresh the outputs.")

top_columns = st.columns(3)
with top_columns[0]:
    show_image_slot("Original", result.original)
with top_columns[1]:
    show_image_slot("Depth Map", result.depth_map)
    show_download("Download depth map", result.depth_map, "depth-map.png")
with top_columns[2]:
    show_image_slot("Depth-Aware Subject", result.subject)
    show_download("Download subject", result.subject, "depth-aware-subject.png")

second_columns = st.columns(3)
with second_columns[0]:
    show_image_slot("Portrait Bokeh", result.bokeh)
    show_download("Download portrait bokeh", result.bokeh, "portrait-bokeh.png")
with second_columns[1]:
    show_caption(result)
with second_columns[2]:
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
