import os
import html

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from transformers import SegformerForSemanticSegmentation


# ---------------------------------------------------------------------
# Model configuration — kept consistent with the original Colab notebook
# ---------------------------------------------------------------------
CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "Traffic light", "Traffic sign", "Vegetation",
    "Terrain", "Sky", "Person", "Rider", "Car",
    "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
]

PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170, 30], [220, 220, 0], [107, 142, 35],
    [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230],
    [119, 11, 32]
], dtype=np.uint8)

CHECKPOINT_PATH = os.environ.get(
    "SEGMENTATION_CHECKPOINT",
    "image_segmentation_epoch10.pth"
)

st.set_page_config(
    page_title="RoadVision AI | Semantic Segmentation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# Custom visual design
# ---------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --bg: #08111f;
    --panel: #101d2e;
    --panel-2: #14243a;
    --line: rgba(148, 163, 184, .17);
    --muted: #9aabc0;
    --text: #edf4fc;
    --cyan: #39d6d0;
    --blue: #6c9cff;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background:
      radial-gradient(circle at 12% 0%, rgba(42, 113, 160, .20), transparent 30%),
      radial-gradient(circle at 90% 8%, rgba(35, 190, 177, .10), transparent 25%),
      var(--bg);
    color: var(--text);
}
.block-container {
    max-width: 1440px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
[data-testid="stSidebar"] {
    background: #0b1727;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: #dce8f6; }
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.035em;
}
p, label, .stCaption { color: #b7c5d6; }
.hero {
    padding: 1.55rem 1.7rem;
    border: 1px solid rgba(110, 170, 210, .22);
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(19, 44, 68, .96), rgba(13, 31, 49, .88));
    box-shadow: 0 18px 55px rgba(0,0,0,.20);
    margin-bottom: 1.25rem;
}
.eyebrow {
    color: var(--cyan);
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: .16em;
    text-transform: uppercase;
}
.hero-title {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(1.8rem, 3.4vw, 2.8rem);
    line-height: 1.12;
    font-weight: 800;
    color: #f5f9ff;
    margin: .45rem 0 .55rem;
}
.hero-sub {
    color: #b7c9dc;
    font-size: 1rem;
    max-width: 850px;
}
.metric-card {\n    height: 100%;
    background: linear-gradient(145deg, rgba(22, 42, 64, .98), rgba(14, 29, 47, .98));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    min-height: 108px;
}
.metric-label {
    color: #91a7bd;
    font-size: .76rem;
    text-transform: uppercase;
    letter-spacing: .10em;
    font-weight: 700;
}
.metric-value {
    color: #f2f7ff;
    font-family: 'Manrope', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: .3rem;
}
.metric-note { color: #8fa5ba; font-size: .78rem; margin-top: .1rem; }
.section-head {
    font-family: 'Manrope', sans-serif;
    color: #edf4fc;
    font-size: 1.25rem;
    font-weight: 800;
    margin: 1.55rem 0 .75rem;
}
.panel {
    background: rgba(16, 29, 46, .82);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem;
}
.panel-label {\n    letter-spacing: .035em;
    color: #c9d8e8;
    font-weight: 700;
    font-size: .92rem;
    margin-bottom: .65rem;
}
.small-muted { color: #91a5ba; font-size: .82rem; }
div.stButton > button {
    border: 0;
    border-radius: 11px;
    min-height: 2.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #2bc9c3, #5b9cff);
    color: #061421;
    box-shadow: 0 8px 22px rgba(43, 201, 195, .16);
}
div.stButton > button:hover {
    border: 0;
    color: #061421;
    filter: brightness(1.08);
}
[data-testid="stFileUploader"] {
    background: rgba(17, 35, 55, .78);
    border: 1px dashed rgba(89, 180, 203, .48);
    border-radius: 15px;
    padding: .6rem;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}
hr { border-color: var(--line); }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
.status-pill {
    display: inline-block;
    border: 1px solid rgba(57,214,208,.28);
    background: rgba(57,214,208,.10);
    color: #74e7df;
    border-radius: 999px;
    padding: .32rem .7rem;
    font-size: .78rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Model loading — preserves the trained checkpoint and 19-class head
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=19,
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, device, checkpoint


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚘 RoadVision AI")
    st.markdown("<span class='status-pill'>SEMANTIC PERCEPTION</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Project details")
    st.markdown("**Architecture**  \nSegFormer-B0")
    st.markdown("**Dataset**  \nCityscapes-related road scenes")
    st.markdown("**Classes**  \n19 semantic categories")
    st.markdown("**Framework**  \nPyTorch · Hugging Face · Streamlit")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Upload a road-scene image.  \n"
        "2. Select **Run segmentation**.  \n"
        "3. Compare the original, mask and overlay.  \n"
        "4. Review the pixel distribution."
    )
    st.markdown("---")
    st.caption("Academic prototype · Image-based inference")


# ---------------------------------------------------------------------
# Hero and model status
# ---------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Computer Vision · Autonomous Driving</div>
  <div class="hero-title">Road Scene Understanding</div>
  <div class="hero-sub">
    Pixel-level scene interpretation powered by a trained SegFormer-B0 model.
    Explore road environments through semantic masks, visual overlays and
    category-wise pixel distribution.
  </div>
</div>
""", unsafe_allow_html=True)

if not os.path.isfile(CHECKPOINT_PATH):
    st.error(
        f"Model checkpoint not found: `{CHECKPOINT_PATH}`. "
        "Place `image_segmentation_epoch10.pth` beside `app.py`, "
        "or set SEGMENTATION_CHECKPOINT to its full path."
    )
    st.stop()

try:
    with st.spinner("Initializing SegFormer-B0 and loading trained weights..."):
        model, device, checkpoint = load_model(CHECKPOINT_PATH)
    epoch = checkpoint.get("epoch", "—")
    pixel_accuracy = checkpoint.get("pixel_accuracy")
    mean_iou = checkpoint.get("mean_iou")
except Exception as exc:
    st.error(f"Could not load the trained checkpoint: {exc}")
    st.stop()

metric_cols = st.columns(4)
metrics = [
    ("MODEL", "SegFormer-B0", "Transformer-based segmentation"),
    ("TRAINING", f"Epoch {epoch}", "Saved training checkpoint"),
    ("CLASSES", "19", "Cityscapes train-ID categories"),
    ("DEVICE", device.type.upper(), "Current inference device")
]
for col, (label, value, note) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""<div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-note">{note}</div>
            </div>""",
            unsafe_allow_html=True
        )

if pixel_accuracy is not None or mean_iou is not None:
    acc_text = f"{float(pixel_accuracy):.2f}%" if pixel_accuracy is not None else "Not recorded"
    iou_text = f"{float(mean_iou):.2f}%" if mean_iou is not None else "Not recorded"
    st.caption(
        f"Checkpoint training metadata — Pixel accuracy: {acc_text} · "
        f"Mean IoU: {iou_text}. These are saved checkpoint metrics, not per-upload scores."
    )

st.markdown("<div class='section-head'>01 / Upload road-scene image</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='small-muted'>Choose a JPG or PNG image. The model processes the image "
    "at 256 × 256 pixels, matching the preprocessing used in the training notebook.</div>",
    unsafe_allow_html=True
)
uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown(
        """<div class="panel" style="margin-top:.8rem;">
          <div class="panel-label">Ready for analysis</div>
          <div class="small-muted">Your uploaded road image and segmentation results will appear here.</div>
        </div>""",
        unsafe_allow_html=True
    )
    st.stop()

try:
    original = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

preview_col, info_col = st.columns([1.6, 1], gap="large")
with preview_col:
    st.image(original, caption="Input road-scene image", width="stretch")
with info_col:
    with st.container(border=True):
        st.markdown("### Image information")
        st.write(f"**File:** {html.escape(uploaded.name)}")
        st.write(f"**Dimensions:** {original.width} × {original.height} px")
        st.write("**Format:** RGB")
        st.write("**Inference size:** 256 × 256 px")
        st.caption("The uploaded image is resized for model inference; result visualizations are displayed at the original dimensions.")

run_col, _ = st.columns([1, 2])
with run_col:
    run_clicked = st.button("▶  Run semantic segmentation", type="primary", width="stretch")

if run_clicked:
    with st.spinner("Analyzing image and generating semantic classes..."):
        # Keep the exact preprocessing used by the original application/notebook.
        input_image = original.resize((256, 256), Image.Resampling.BILINEAR)
        image_array = np.asarray(input_image, dtype=np.float32) / 255.0
        input_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        with torch.inference_mode():
            outputs = model(pixel_values=input_tensor)
            logits = outputs.logits
            logits = F.interpolate(
                logits,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )
            prediction = logits.argmax(dim=1)[0].cpu().numpy()

        color_mask = PALETTE[prediction]
        mask_image = Image.fromarray(color_mask).resize(
            original.size,
            Image.Resampling.NEAREST
        )
        overlay = Image.blend(original, mask_image.convert("RGB"), alpha=0.45)

        unique_ids, counts = np.unique(prediction, return_counts=True)
        total = int(counts.sum())
        present = sorted(
            zip(unique_ids.tolist(), counts.tolist()),
            key=lambda item: item[1],
            reverse=True
        )

    st.markdown("<div class='section-head'>02 / Segmentation results</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-muted'>Compare the input image, predicted semantic mask and blended overlay.</div>",
        unsafe_allow_html=True
    )

    result_cols = st.columns(3, gap="large")
    result_items = [
        (result_cols[0], "01 · ORIGINAL IMAGE", original),
        (result_cols[1], "02 · PREDICTED MASK", mask_image),
        (result_cols[2], "03 · OVERLAY VIEW", overlay)
    ]
    for col, label, result_image in result_items:
        with col:
            with st.container(border=True):
                st.markdown(f"<div class='panel-label'>{label}</div>", unsafe_allow_html=True)
                st.image(result_image, width="stretch")

    st.markdown("<div class='section-head'>03 / Semantic category analysis</div>", unsafe_allow_html=True)
    rows = [
        {
            "Category": CLASS_NAMES[class_id],
            "Pixel count": int(pixel_count),
            "Pixel share (%)": round(pixel_count / total * 100, 2)
        }
        for class_id, pixel_count in present
    ]
    df = pd.DataFrame(rows)

    analysis_left, analysis_right = st.columns([1.15, 1], gap="large")
    with analysis_left:
        st.markdown("<div class='panel-label'>Detected categories</div>", unsafe_allow_html=True)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            height=min(420, 42 + 38 * len(df))
        )
    with analysis_right:
        st.markdown("<div class='panel-label'>Pixel distribution</div>", unsafe_allow_html=True)
        chart_df = df.set_index("Category")[["Pixel share (%)"]]
        st.bar_chart(chart_df, width="stretch", height=360)

    st.caption(
        "Pixel share is the proportion of image pixels assigned to each predicted class. "
        "It is not an object count or a confidence score."
    )

st.markdown("---")
st.markdown(
    """
    <div style="padding:1.1rem 0 .5rem;text-align:center;">
      <div style="color:#53d9d2;font-weight:800;letter-spacing:.13em;font-size:.75rem;">
        ROADVISION AI
      </div>
      <div style="color:#c2d0df;font-size:.88rem;margin-top:.35rem;">
        SegFormer-B0 · 19-class semantic segmentation · Road-scene understanding
      </div>
      <div style="color:#8297ad;font-size:.76rem;margin-top:.35rem;">
        Academic computer-vision prototype
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
