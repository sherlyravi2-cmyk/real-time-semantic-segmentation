import os
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

# Must match the notebook's 19 Cityscapes train-ID classes.
CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "Traffic light", "Traffic sign", "Vegetation",
    "Terrain", "Sky", "Person", "Rider", "Car",
    "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
]

# Cityscapes RGB palette, in the same class order as CLASS_NAMES.
PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153],
    [250, 170, 30], [220, 220, 0], [107, 142, 35],
    [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230],
    [119, 11, 32]
], dtype=np.uint8)

st.set_page_config(
    page_title="Road Scene Semantic Segmentation",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Real-Time Road Scene Semantic Segmentation")
st.write(
    "Upload a road-scene image to classify its pixels into "
    "19 Cityscapes categories using the trained SegFormer-B0 model."
)

checkpoint_path = os.environ.get(
    "SEGMENTATION_CHECKPOINT",
    "image_segmentation_epoch10.pth"
)

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match the model family and 19-class head used in the Colab notebook.
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

if not os.path.isfile(checkpoint_path):
    st.error(
        f"Checkpoint not found: {checkpoint_path}. "
        "Place image_segmentation_epoch10.pth beside app.py, "
        "or set SEGMENTATION_CHECKPOINT to its full path."
    )
    st.stop()

try:
    with st.spinner("Loading the trained segmentation model..."):
        model, device, checkpoint = load_model(checkpoint_path)
    st.success(
        f"Model loaded · Epoch {checkpoint.get('epoch', 'unknown')} · "
        f"Running on {device.type.upper()}"
    )
except Exception as exc:
    st.error(f"Could not load the model checkpoint: {exc}")
    st.stop()

uploaded = st.file_uploader(
    "Choose a road-scene image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    original = Image.open(uploaded).convert("RGB")
    st.image(original, caption="Uploaded image", use_container_width=True)

    if st.button("Run segmentation", type="primary"):
        with st.spinner("Segmenting the image..."):
            # Notebook training used 256x256 half-images and raw [0,1]
            # pixel scaling. Keep that preprocessing consistent.
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

            # Convert class IDs to the Cityscapes display palette.
            color_mask = PALETTE[prediction]

            # Resize display mask back to the uploaded image dimensions.
            mask_image = Image.fromarray(color_mask).resize(
                original.size, Image.Resampling.NEAREST
            )

            overlay = Image.blend(
                original,
                mask_image.convert("RGB"),
                alpha=0.45
            )

            unique_ids, counts = np.unique(prediction, return_counts=True)
            total = counts.sum()
            present = sorted(
                zip(unique_ids.tolist(), counts.tolist()),
                key=lambda item: item[1],
                reverse=True
            )

        st.subheader("Segmentation output")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption="Original image", use_container_width=True)
        with col2:
            st.image(mask_image, caption="Predicted class mask", use_container_width=True)
        with col3:
            st.image(overlay, caption="Overlay", use_container_width=True)

        st.subheader("Detected categories")
        rows = [
            {
                "Category": CLASS_NAMES[class_id],
                "Pixel share (%)": round(pixel_count / total * 100, 2)
            }
            for class_id, pixel_count in present
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.caption(
            "Pixel share describes the proportion of image pixels assigned "
            "to each class. It is not an object count or a confidence score."
        )
else:
    st.info("Upload a road-scene image to begin.")
