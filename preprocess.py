"""
preprocess_and_balance.py — Full pipeline:
  1. Preprocess ALL images (resize, CLAHE, Ben Graham)
  2. Augment ONLY rare classes (1_Mild, 3_Severe, 4_PDR) up to TARGET_COUNT
  3. Classes 0_No_DR and 2_Moderate are preprocessed as-is (no touching)

Output saved to a SEPARATE folder — original dataset is untouched.

Install deps:
    pip install albumentations opencv-python pillow tqdm numpy

Usage:
    python preprocess_and_balance.py
"""

import os
import random
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import albumentations as A

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

INPUT_DIR  = r"C:\Users\xiaon\Downloads\dr ai\project\dataset\train"
OUTPUT_DIR = r"C:\Users\xiaon\Downloads\dr ai\project\dataset\balanced"

IMAGE_SIZE   = 512
TARGET_COUNT = 1400        # Rare classes will be augmented up to this
RARE_CLASSES = {'1_Mild', '3_Severe', '4_PDR'}   # Only these get augmented
IMG_EXTS     = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

CLASS_MAP = ["0_No_DR", "1_Mild", "2_Moderate", "3_Severe", "4_PDR"]

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (applied to EVERY image regardless of class)
# ═══════════════════════════════════════════════════════════════════════

def remove_black_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask   = gray > tol
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def apply_clahe(img: np.ndarray) -> np.ndarray:
    lab     = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    lab     = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def ben_graham_preprocessing(img: np.ndarray, size: int) -> np.ndarray:
    h, w  = img.shape[:2]
    scale = size / min(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img   = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    start_y = (new_h - size) // 2
    start_x = (new_w - size) // 2
    img = img[start_y:start_y + size, start_x:start_x + size]

    blurred = cv2.GaussianBlur(img, (0, 0), size / 30)
    img     = cv2.addWeighted(img, 4, blurred, -4, 128)
    img     = np.clip(img, 0, 255).astype(np.uint8)
    return img


def preprocess(path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8)
    img = remove_black_border(img)
    img = apply_clahe(img)
    img = ben_graham_preprocessing(img, size)
    return img


# ═══════════════════════════════════════════════════════════════════════
#  AUGMENTATION  (only for rare classes)
# ═══════════════════════════════════════════════════════════════════════

aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),

    A.ShiftScaleRotate(
        shift_limit=0.08,
        scale_limit=0.12,
        rotate_limit=30,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.8
    ),

    A.OneOf([
        A.ElasticTransform(alpha=80, sigma=8, p=1.0),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
        A.OpticalDistortion(distort_limit=0.2, p=1.0),
    ], p=0.4),

    A.OneOf([
        A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=1.0),
    ], p=0.65),

    A.OneOf([
        A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.25), p=1.0),
    ], p=0.3),

    A.CoarseDropout(
        num_holes_range=(1, 3),
        hole_height_range=(8, 20),
        hole_width_range=(8, 20),
        fill=0,
        p=0.15
    ),
])


def augment(img: np.ndarray) -> np.ndarray:
    return aug_transform(image=img)['image']


# ═══════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_images(cls_dir: str):
    return [
        f for f in os.listdir(cls_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ]


def save(img: np.ndarray, path: str):
    Image.fromarray(img).save(path)


def bar(n: int, ref: int = 2239, width: int = 40) -> str:
    filled = int(width * n / max(ref, 1))
    return '█' * filled + '░' * (width - filled)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  DR DATASET — PREPROCESS + AUGMENT RARE CLASSES PIPELINE")
print(f"  Image size       : {IMAGE_SIZE}×{IMAGE_SIZE}")
print(f"  Rare target      : {TARGET_COUNT} (classes 1, 3, 4 only)")
print(f"  Output           : {OUTPUT_DIR}")
print("=" * 65)

for cls_name in CLASS_MAP:
    os.makedirs(os.path.join(OUTPUT_DIR, cls_name), exist_ok=True)

# ── Show input state ─────────────────────────────────────────────────
print("\nInput distribution:")
for cls_name in CLASS_MAP:
    src_dir = os.path.join(INPUT_DIR, cls_name)
    count   = len(get_images(src_dir)) if os.path.isdir(src_dir) else 0
    tag     = "  ← will augment" if cls_name in RARE_CLASSES else ""
    print(f"  {cls_name:20s}: {count:5d}  {bar(count)}{tag}")

print()

# ═══════════════════════════════════════════════════════════════════════
#  PROCESS EACH CLASS
# ═══════════════════════════════════════════════════════════════════════

for cls_name in CLASS_MAP:
    src_dir = os.path.join(INPUT_DIR, cls_name)
    dst_dir = os.path.join(OUTPUT_DIR, cls_name)

    if not os.path.isdir(src_dir):
        print(f"[WARN] {src_dir} not found — skipping.")
        continue

    all_files = get_images(src_dir)
    current   = len(all_files)
    is_rare   = cls_name in RARE_CLASSES

    print(f"\n{'─'*65}")
    if is_rare:
        print(f"  CLASS: {cls_name}  ({current} → {TARGET_COUNT})  [PREPROCESS + AUGMENT]")
    else:
        print(f"  CLASS: {cls_name}  ({current})  [PREPROCESS ONLY]")
    print(f"{'─'*65}")

    # ── Step 1: Preprocess ALL images ───────────────────────────────
    preprocessed_paths = []
    print(f"  Preprocessing {current} images …")

    for fname in tqdm(all_files, desc=f"  Preprocess {cls_name}", unit="img", leave=False):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        try:
            img = preprocess(src_path)
            save(img, dst_path)
            preprocessed_paths.append(dst_path)
        except Exception as e:
            print(f"\n  [WARN] Skipped {fname}: {e}")

    saved = len(preprocessed_paths)
    print(f"  Preprocessed   : {saved} images saved ✓")

    # ── Step 2: Augment rare classes only ───────────────────────────
    if not is_rare:
        print(f"  Augmentation   : skipped (keeping original count)")
        continue

    needed = TARGET_COUNT - saved
    if needed <= 0:
        print(f"  Augmentation   : not needed (already ≥ {TARGET_COUNT})")
        continue

    print(f"  Augmenting     : +{needed} images needed …")
    counter   = 0
    generated = 0

    pbar = tqdm(total=needed, desc=f"  Augment  {cls_name}", unit="img", leave=False)
    while generated < needed:
        src_path = random.choice(preprocessed_paths)
        try:
            img     = np.array(Image.open(src_path).convert('RGB'), dtype=np.uint8)
            aug_img = augment(img)
            fname   = os.path.basename(src_path)
            out_name = f"aug_{counter:05d}_{fname}"
            save(aug_img, os.path.join(dst_dir, out_name))
            counter   += 1
            generated += 1
            pbar.update(1)
        except Exception as e:
            print(f"\n  [WARN] Aug failed: {e}")
    pbar.close()
    print(f"  Augmented      : +{generated} images saved ✓")

# ═══════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  FINAL DISTRIBUTION")
print("=" * 65)
grand_total = 0
ref = max(
    len(os.listdir(os.path.join(OUTPUT_DIR, c)))
    for c in CLASS_MAP if os.path.isdir(os.path.join(OUTPUT_DIR, c))
)
for cls_name in CLASS_MAP:
    dst_dir = os.path.join(OUTPUT_DIR, cls_name)
    count   = len(os.listdir(dst_dir)) if os.path.isdir(dst_dir) else 0
    grand_total += count
    print(f"  {cls_name:20s}: {count:5d}  {bar(count, ref=ref)}")

print(f"\n  Grand total    : {grand_total} images")
print(f"  Output folder  : {OUTPUT_DIR}")
print("=" * 65)
print("✅ Done! Point train_kfold.py to the balanced folder.")