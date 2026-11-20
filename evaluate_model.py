import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SegformerForSemanticSegmentation


# ==========================================
# 1. CONFIGURATION
# ==========================================

CHECKPOINT_PATH = os.path.expanduser(
    "~/Downloads/Image_Segmentation_Prototype/image_segmentation_epoch10.pth"
)

TRAIN_PATH = os.path.expanduser(
    "~/Desktop/cityscapes_data/cityscapes_data/train"
)

NUM_CLASSES = 19
IGNORE_INDEX = 255
THRESHOLD = 35
BATCH_SIZE = 4
RANDOM_SEED = 42

CLASS_NAMES = [
    "Road", "Sidewalk", "Building", "Wall", "Fence",
    "Pole", "Traffic light", "Traffic sign", "Vegetation",
    "Terrain", "Sky", "Person", "Rider", "Car",
    "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
]

PALETTE = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]
], dtype=np.float32)


# ==========================================
# 2. DEVICE AND FILE CHECKS
# ==========================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 55)
print("SEGFORMER EPOCH 10 MODEL EVALUATION")
print("=" * 55)

print("Device:", DEVICE)

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

if not os.path.isdir(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_PATH}"
    )


# ==========================================
# 3. RECREATE THE ORIGINAL DATASET
# ==========================================

class CityscapesDataset(Dataset):

    def __init__(self, folder, palette, threshold=35):

        self.folder = folder
        self.palette = np.asarray(palette, dtype=np.float32)
        self.threshold = threshold

        self.files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filepath = os.path.join(
            self.folder,
            self.files[idx]
        )

        combined = np.array(
            Image.open(filepath).convert("RGB")
        )

        h, w, _ = combined.shape

        # Left half: original road image
        road_image = combined[:, :w // 2, :]

        # Right half: segmentation mask
        mask_rgb = combined[:, w // 2:, :]

        # Convert RGB mask colours to nearest class IDs
        mask_float = mask_rgb.astype(np.float32)

        distances = np.sqrt(
            (
                mask_float[:, :, None, :]
                - self.palette[None, None, :, :]
            ) ** 2
        ).sum(axis=3)

        distances = np.sqrt(
            (
                mask_float[:, :, None, :]
                - self.palette[None, None, :, :]
            ) ** 2
        ).sum(axis=3)

        nearest_distance = distances.min(axis=2)
        nearest_class = distances.argmin(axis=2)

        label_mask = np.full(
            mask_rgb.shape[:2],
            IGNORE_INDEX,
            dtype=np.uint8
        )

        valid_pixels = nearest_distance < self.threshold

        label_mask[valid_pixels] = nearest_class[valid_pixels]

        image_tensor = torch.from_numpy(
            road_image.astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        mask_tensor = torch.from_numpy(
            label_mask.astype(np.int64)
        )

        return image_tensor, mask_tensor


# ==========================================
# 4. RECREATE THE ORIGINAL 80/20 SPLIT
# ==========================================

dataset = CityscapesDataset(
    TRAIN_PATH,
    PALETTE,
    threshold=THRESHOLD
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nDATASET INFORMATION")
print("Total images:", len(dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Validation batches:", len(val_loader))


# ==========================================
# 5. LOAD THE EPOCH 10 CHECKPOINT
# ==========================================

print("\nLoading Epoch 10 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

if checkpoint.get("epoch") != 10:
    raise ValueError(
        "The selected checkpoint is not Epoch 10."
    )

if "model_state_dict" not in checkpoint:
    raise KeyError(
        "model_state_dict is missing from the checkpoint."
    )

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)
model.eval()

print("Epoch:", checkpoint["epoch"])
print("Checkpoint loaded successfully!")


# ==========================================
# 6. VALIDATION AND CONFUSION MATRIX
# ==========================================

confusion_matrix = torch.zeros(
    NUM_CLASSES,
    NUM_CLASSES,
    dtype=torch.int64
)

print("\nStarting validation...")

with torch.inference_mode():

    for batch_number, (images, masks) in enumerate(val_loader, 1):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long()

        outputs = model(pixel_values=images)

        logits = F.interpolate(
            outputs.logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predictions = logits.argmax(dim=1)

        valid = (
            (masks != IGNORE_INDEX)
            & (masks >= 0)
            & (masks < NUM_CLASSES)
        )

        targets = masks[valid]
        preds = predictions[valid]

        indices = (
            targets * NUM_CLASSES + preds
        )

        batch_confusion = torch.bincount(
            indices,
            minlength=NUM_CLASSES * NUM_CLASSES
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        confusion_matrix += batch_confusion.cpu()

        if batch_number % 20 == 0 or batch_number == len(val_loader):
            print(
                f"Processed {batch_number}/{len(val_loader)} batches"
            )


# ==========================================
# 7. CALCULATE PIXEL ACCURACY AND mIoU
# ==========================================

confusion = confusion_matrix.float()

intersection = torch.diag(confusion)

ground_truth_pixels = confusion.sum(dim=1)

predicted_pixels = confusion.sum(dim=0)

union = (
    ground_truth_pixels
    + predicted_pixels
    - intersection
)

valid_classes = union > 0

iou_per_class = (
    intersection[valid_classes]
    / union[valid_classes]
)

mean_iou = iou_per_class.mean()

total_pixels = confusion.sum()

pixel_accuracy = (
    intersection.sum() / total_pixels
)


# ==========================================
# 8. DISPLAY FINAL RESULTS
# ==========================================

print("\n" + "=" * 55)
print("EPOCH 10 VALIDATION RESULTS")
print("=" * 55)

print("Epoch:", checkpoint["epoch"])

print(
    "Pixel Accuracy:",
    round(pixel_accuracy.item() * 100, 4),
    "%"
)

print(
    "Mean IoU:",
    round(mean_iou.item() * 100, 4),
    "%"
)

print(
    "Classes evaluated:",
    valid_classes.sum().item()
)

print("\nPER-CLASS IoU")
print("-" * 40)

class_index = 0

for i in range(NUM_CLASSES):

    if valid_classes[i]:

        print(
            f"{CLASS_NAMES[i]:18s}: "
            f"{iou_per_class[class_index].item() * 100:.4f}%"
        )

        class_index += 1

    else:

        print(
            f"{CLASS_NAMES[i]:18s}: No valid pixels"
        )


# ==========================================
# 9. COMPARE WITH SAVED CHECKPOINT METRICS
# ==========================================

print("\nSAVED CHECKPOINT METRICS")
print("-" * 40)

if "pixel_accuracy" in checkpoint:
    print(
        "Saved Pixel Accuracy:",
        checkpoint["pixel_accuracy"],
        "%"
    )

if "mean_iou" in checkpoint:
    print(
        "Saved mIoU:",
        checkpoint["mean_iou"],
        "%"
    )

print("\nEVALUATION COMPLETED SUCCESSFULLY!")
print("Original checkpoint and app.py were not modified.")
