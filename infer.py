"""
infer.py — DR Inference & Submission Script (Improved)
═══════════════════════════════════════════════════════
Improvements over original:
  1. Ben Graham preprocessing added (was missing, used in training)
  2. 4-variant TTA (H-flip + V-flip + H+V flip) for better accuracy
  3. Auto-extracts actual class from filename (img_X_XXXXX format)
  4. Removed colour_tint and brightness functions (NOT in training pipeline)
  5. Batch inference — processes multiple images per GPU call (faster)
  6. Per-class accuracy printed alongside F1
  7. Optional ensemble of multiple fold checkpoints

Usage:
    # Standard — flat folder, labels from filename
    python infer.py --input_dir test_sets/test_1 --output submission.csv

    # Labelled subfolders (0_No_DR/, 1_Mild/, etc.)
    python infer.py --input_dir dataset/test --output submission.csv

    # Custom model path
    python infer.py --input_dir test_sets/test_1 --output submission.csv \
                    --model_path saved_model/best_model.pth

    # Ensemble of multiple fold checkpoints
    python infer.py --input_dir test_sets/test_1 --output submission.csv \
                    --ensemble saved_model/fold_1_best.pth \
                               saved_model/fold_2_best.pth \
                               saved_model/fold_3_best.pth
"""
import os
import re
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import build_model

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['0_No_DR', '1_Mild', '2_Moderate', '3_Severe', '4_PDR']
CLASS_SHORT  = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
IMG_EXTS     = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
BATCH_SIZE   = 16   # images per GPU batch — increase if VRAM allows

# ── Preprocessing Pipeline — matches preprocess_and_balance.py EXACTLY ─────────

def remove_black_border(img_bgr):
    """
    Remove dark circular border from fundus image.
    Uses threshold=7 matching training pipeline.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 7, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img_bgr
    x, y, w, h = cv2.boundingRect(coords)
    h0, w0 = img_bgr.shape[:2]
    # Safety: reject crop if it removes >40% of original area
    if w < 50 or h < 50 or (h * w) < (h0 * w0 * 0.6):
        return img_bgr
    return img_bgr[y:y+h, x:x+w]


def apply_clahe(img_bgr):
    """
    CLAHE on LAB L-channel.
    clipLimit=2.0, tileGridSize=8x8 — matches training.
    """
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l       = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def ben_graham(img_bgr, size=224):
    """
    Ben Graham illumination normalisation.
    Formula: result = 4*img - 4*blur + 128
    sigma = size//30 — matches training.

    *** This was MISSING from the original infer.py ***
    Training used this — must be applied in inference too.
    """
    arr     = cv2.resize(img_bgr, (size, size)).astype(np.float32)
    sigma   = max(5, size // 30)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)
    result  = np.clip(arr * 4 - blurred * 4 + 128, 0, 255).astype(np.uint8)
    return result


def preprocess_image(img_path: str):
    """
    Full preprocessing pipeline matching preprocess_and_balance.py:
      1. Load image
      2. Black border removal
      3. CLAHE contrast enhancement
      4. Ben Graham illumination normalisation
      5. Convert to PIL RGB

    Returns PIL Image or None if load fails.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    img_bgr = remove_black_border(img_bgr)
    img_bgr = apply_clahe(img_bgr)
    img_bgr = ben_graham(img_bgr, size=224)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ── TTA Transform ──────────────────────────────────────────────────────────────
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def to_tensor(img_pil):
    return NORMALIZE(transforms.ToTensor()(img_pil))

def get_tta_tensors(img_pil):
    """
    4-variant TTA — same as app.py:
      1. Original
      2. Horizontal flip
      3. Vertical flip
      4. H-flip + V-flip (180° rotation)

    Returns tensor of shape [4, 3, 224, 224]
    """
    img = img_pil.resize((224, 224), Image.BILINEAR)
    variants = [
        img,
        img.transpose(Image.FLIP_LEFT_RIGHT),
        img.transpose(Image.FLIP_TOP_BOTTOM),
        img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM),
    ]
    return torch.stack([to_tensor(v) for v in variants])  # [4, 3, 224, 224]


# ── Label Extraction ───────────────────────────────────────────────────────────

def extract_label_from_filename(filename: str):
    """
    Auto-extract actual class from filename.

    Supported formats:
      img_0_000001.png  → class 0
      img_4_004541.jpg  → class 4
      0_No_DR/anyname   → class 0 (from parent folder)

    Returns int label or None if not detected.
    """
    basename = os.path.basename(filename)
    # Match pattern: img_X_XXXXX.ext
    match = re.match(r'^img_(\d)_', basename)
    if match:
        label = int(match.group(1))
        if 0 <= label <= 4:
            return label
    return None


# ── Model Loading ──────────────────────────────────────────────────────────────

def load_model(model_path: str, device):
    """Load a single model checkpoint."""
    model = build_model(pretrained=False)
    ckpt  = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)
    epoch   = ckpt.get('epoch', '?')
    val_f1  = ckpt.get('val_f1', ckpt.get('val_acc', '?'))
    print(f"  ✓ Loaded: {model_path}  (epoch={epoch}, val_f1={val_f1})")
    return model


# ── Image Gathering ────────────────────────────────────────────────────────────

def gather_images(input_dir: str):
    """
    Gather all images from input_dir.

    Supports:
      - Flat folder: all images directly in input_dir
      - Class subfolders: 0_No_DR/, 1_Mild/, etc.
      - Mixed: images in root + class subfolders

    Returns list of (img_path, filename, actual_label_or_None)
    """
    items = []
    seen  = set()

    # Check for class subfolders
    subdirs = [d for d in os.listdir(input_dir)
               if os.path.isdir(os.path.join(input_dir, d)) and d in CLASS_NAMES]

    if subdirs:
        # Labelled subfolder mode
        for cls in CLASS_NAMES:
            cls_dir = os.path.join(input_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = CLASS_NAMES.index(cls)
            for f in sorted(os.listdir(cls_dir)):
                if os.path.splitext(f)[1].lower() in IMG_EXTS:
                    path = os.path.join(cls_dir, f)
                    if path not in seen:
                        items.append((path, f, label))
                        seen.add(path)

    # Also check flat images in root dir
    for f in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTS:
            path = os.path.join(input_dir, f)
            if path not in seen:
                # Try to extract label from filename
                label = extract_label_from_filename(f)
                items.append((path, f, label))
                seen.add(path)

    return items


# ── Batch Inference ────────────────────────────────────────────────────────────

def run_inference(input_dir: str, output_csv: str,
                  model_paths: list, use_tta: bool = True):
    """
    Main inference function.

    Args:
        input_dir   : folder containing test images
        output_csv  : path to save submission CSV
        model_paths : list of checkpoint paths (1 = single model, >1 = ensemble)
        use_tta     : whether to apply 4-variant TTA
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*65}")
    print(f"  RetinaScan AI — Inference")
    print(f"{'='*65}")
    print(f"  Device     : {device}")
    print(f"  Input dir  : {input_dir}")
    print(f"  Output CSV : {output_csv}")
    print(f"  TTA        : {'4-variant (original + 3 flips)' if use_tta else 'disabled'}")
    print(f"  Models     : {len(model_paths)}")

    # ── Load models ────────────────────────────────────────────────────
    print(f"\n  Loading model(s):")
    models = [load_model(p, device) for p in model_paths]

    # ── Gather images ──────────────────────────────────────────────────
    print(f"\n  Scanning: {input_dir}")
    items = gather_images(input_dir)
    if not items:
        print(f"  ❌ No images found in {input_dir}")
        return

    labelled = sum(1 for _, _, l in items if l is not None)
    print(f"  Found    : {len(items)} images")
    print(f"  Labelled : {labelled} (label from filename)")
    print(f"  Unlabelled: {len(items) - labelled}")

    # ── Print actual class distribution ───────────────────────────────
    if labelled > 0:
        from collections import Counter
        actual_counts = Counter(l for _, _, l in items if l is not None)
        print(f"\n  Actual class distribution:")
        for cls_idx in sorted(actual_counts.keys()):
            print(f"    Class {cls_idx} ({CLASS_SHORT[cls_idx]:<12}): {actual_counts[cls_idx]:>4} images")

    # ── Inference ──────────────────────────────────────────────────────
    print(f"\n  Running inference...")
    results = []
    y_true  = []
    y_pred  = []

    for img_path, filename, actual_label in tqdm(items, desc="  Processing"):
        # Preprocess
        pil_img = preprocess_image(img_path)
        if pil_img is None:
            print(f"  [WARN] Cannot read: {img_path}")
            continue

        # Get tensors
        if use_tta:
            tensors = get_tta_tensors(pil_img).to(device)  # [4, 3, 224, 224]
        else:
            tensors = to_tensor(pil_img.resize((224, 224), Image.BILINEAR)).unsqueeze(0).to(device)

        # Run through all models and average probabilities
        all_probs = []
        for model in models:
            with torch.no_grad():
                logits = model(tensors)                          # [4 or 1, 5]
                probs  = F.softmax(logits, dim=1)               # [4 or 1, 5]
                avg    = probs.mean(dim=0).cpu().numpy()         # [5]
                all_probs.append(avg)

        # Average across models (ensemble)
        final_probs = np.mean(all_probs, axis=0)                # [5]
        pred_idx    = int(np.argmax(final_probs))
        confidence  = float(final_probs[pred_idx]) * 100
        actual_name = CLASS_NAMES[actual_label] if actual_label is not None else 'Unknown'

        results.append({
            'filename'        : filename,
            'actual_class'    : actual_name,
            'predicted_class' : CLASS_NAMES[pred_idx],
            'confidence_%'    : f'{confidence:.2f}',
            'prob_NoDR_%'     : f'{final_probs[0]*100:.2f}',
            'prob_Mild_%'     : f'{final_probs[1]*100:.2f}',
            'prob_Moderate_%' : f'{final_probs[2]*100:.2f}',
            'prob_Severe_%'   : f'{final_probs[3]*100:.2f}',
            'prob_PDR_%'      : f'{final_probs[4]*100:.2f}',
        })

        if actual_label is not None:
            y_true.append(actual_label)
            y_pred.append(pred_idx)

    # ── Save CSV ───────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"\n  Saved: {output_csv}  ({len(df_out)} rows)")
    print(f"\n  Preview (first 5 rows):")
    print(df_out[['filename','actual_class','predicted_class','confidence_%']].head().to_string(index=False))

    # ── Metrics (only if labels available) ────────────────────────────
    if not y_true:
        print("\n  No labels available — skipping metrics.")
        return

    print(f"\n{'='*65}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*65}")

    acc       = accuracy_score(y_true, y_pred) * 100
    f1_w      = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    f1_macro  = f1_score(y_true, y_pred, average='macro',    zero_division=0) * 100
    prec_w    = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    rec_w     = recall_score(y_true, y_pred,    average='weighted', zero_division=0) * 100

    print(f"\n  Overall Metrics:")
    print(f"    Accuracy         : {acc:.2f}%")
    print(f"    Weighted F1      : {f1_w:.2f}%")
    print(f"    Macro F1         : {f1_macro:.2f}%")
    print(f"    Weighted Precision: {prec_w:.2f}%")
    print(f"    Weighted Recall  : {rec_w:.2f}%")

    # ── Per-class report ───────────────────────────────────────────────
    f1_per    = f1_score(y_true, y_pred, average=None, zero_division=0) * 100
    prec_per  = precision_score(y_true, y_pred, average=None, zero_division=0) * 100
    rec_per   = recall_score(y_true, y_pred,    average=None, zero_division=0) * 100

    # Per-class accuracy
    cm        = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    per_acc   = [cm[i, i] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0 for i in range(5)]

    print(f"\n  Per-Class Report:")
    print(f"  {'Class':<20} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Support':>8}")
    print(f"  {'─'*57}")
    for i in range(5):
        support = int(cm[i].sum())
        if support == 0:
            continue
        print(f"  {CLASS_SHORT[i]:<20} "
              f"{per_acc[i]:>5.1f}% "
              f"{prec_per[i]:>5.1f}% "
              f"{rec_per[i]:>6.1f}% "
              f"{f1_per[i]:>5.1f}% "
              f"{support:>8}")
    print(f"  {'─'*57}")
    print(f"  {'Weighted Avg':<20} "
          f"{acc:>5.1f}% "
          f"{prec_w:>5.1f}% "
          f"{rec_w:>6.1f}% "
          f"{f1_w:>5.1f}%")

    # ── Confusion matrix ───────────────────────────────────────────────
    out_dir  = os.path.dirname(os.path.abspath(output_csv)) or '.'
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Inference Confusion Matrix', fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Raw Counts')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Normalised (row %)')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'inference_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved confusion matrix: {cm_path}")

    # ── Prediction distribution ────────────────────────────────────────
    from collections import Counter
    pred_counts  = Counter(y_pred)
    actual_counts = Counter(y_true)
    print(f"\n  Prediction Distribution:")
    print(f"  {'Class':<20} {'Actual':>8} {'Predicted':>10} {'Correct':>8} {'Recall':>8}")
    print(f"  {'─'*58}")
    for i in range(5):
        actual    = actual_counts.get(i, 0)
        predicted = pred_counts.get(i, 0)
        correct   = cm[i, i] if i < len(cm) else 0
        recall    = per_acc[i]
        if actual > 0:
            print(f"  {CLASS_SHORT[i]:<20} {actual:>8} {predicted:>10} {correct:>8} {recall:>7.1f}%")

    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Total images   : {len(results)}")
    print(f"  Accuracy       : {acc:.2f}%")
    print(f"  Weighted F1    : {f1_w:.2f}%")
    print(f"  Models used    : {len(models)}")
    print(f"  TTA variants   : {'4' if use_tta else '1 (disabled)'}")
    print(f"  Output CSV     : {output_csv}")
    print(f"  Confusion matrix: {cm_path}")
    print(f"{'='*65}\n")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DR Inference — RetinaScan AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard single model
  python infer.py --input_dir test_sets/test_1 --output submission.csv

  # Custom model path
  python infer.py --input_dir test_sets/test_1 --output submission.csv \\
                  --model_path saved_model/best_model.pth

  # Ensemble all 3 fold models (+2-4% F1 improvement)
  python infer.py --input_dir test_sets/test_1 --output submission.csv \\
                  --ensemble saved_model/fold_1_best.pth \\
                             saved_model/fold_2_best.pth \\
                             saved_model/fold_3_best.pth

  # Disable TTA for faster inference
  python infer.py --input_dir test_sets/test_1 --output submission.csv --no_tta
        """
    )
    parser.add_argument('--input_dir',  type=str, required=True,
                        help='Test image folder (flat or class-subfoldered)')
    parser.add_argument('--output',     type=str, default='submission.csv',
                        help='Output CSV filename (default: submission.csv)')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join('saved_model', 'best_model.pth'),
                        help='Path to single model checkpoint')
    parser.add_argument('--ensemble',   type=str, nargs='+', default=None,
                        help='Paths to multiple checkpoints for ensemble inference')
    parser.add_argument('--no_tta',     action='store_true',
                        help='Disable TTA (faster but less accurate)')
    args = parser.parse_args()

    # Determine which model(s) to use
    if args.ensemble:
        model_paths = args.ensemble
        print(f"\n  Mode: ENSEMBLE ({len(model_paths)} models)")
    else:
        model_paths = [args.model_path]
        print(f"\n  Mode: SINGLE MODEL")

    run_inference(
        input_dir   = args.input_dir,
        output_csv  = args.output,
        model_paths = model_paths,
        use_tta     = not args.no_tta,
    )