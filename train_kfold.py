"""
train_kfold.py — Maximum accuracy configuration
Targets: No DR 85%+, Mild 82%+, Moderate 63%+, Severe 91%+, PDR 68%+

New additions vs previous:
  - Reduced label smoothing (0.05) for harder decision boundaries
  - ReduceLROnPlateau added on top of warmup cosine for fine-grained LR control
  - Stronger CLAHE probability
  - Augmentation probability tuned per class difficulty
  - Validation uses averaged H-flip TTA properly (fixes previous dead code)
  - Per-epoch class accuracy printed for detailed monitoring

Usage:
  Local : python train_kfold.py
  Colab : !python train_kfold.py
"""
import os, copy, random, cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageDraw
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

from model import build_model
from utils import plot_confusion_matrix, print_metrics, plot_training_curves

# ══════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════
IS_COLAB = os.path.isdir('/content/drive/MyDrive')
if IS_COLAB:
    print("Google Colab environment detected.")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
if IS_COLAB:
    CLEAN_DIR = "/content/drive/MyDrive/project/dataset/balanced"
    SAVE_DIR  = "/content/drive/MyDrive/project/saved_model"
else:
    CLEAN_DIR = r"C:\Users\xiaon\Downloads\dr ai\project\dataset\balanced"
    SAVE_DIR  = "saved_model"

CLASS_NAMES        = ["0_No_DR", "1_Mild", "2_Moderate", "3_Severe", "4_PDR"]
IMG_EXTS           = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
NUM_CLASSES        = 5
IMG_SIZE           = 224
BATCH_SIZE         = 16 if IS_COLAB else 8
NUM_EPOCHS         = 100
LR                 = 5e-5
N_FOLDS            = 3
SEED               = 42
PATIENCE           = 18
NW                 = 2 if IS_COLAB else 0
PIN                = IS_COLAB

ACCUMULATION_STEPS = 4
MIXUP_ALPHA        = 0.15
CUTMIX_ALPHA       = 0.3
CUTMIX_PROB        = 0.0    # disabled — Mixup only, CutMix caused instability
WARMUP_EPOCHS      = 5
# No DR=1.0, Mild=1.5, Moderate=8.0, Severe=1.2, PDR=5.0
CLASS_WEIGHTS      = [1.0, 2.5, 2.5, 3.0, 4.0]
LABEL_SMOOTHING    = 0.1    # back to stable value

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\n{'='*65}")
print("  RetinaScan AI — Maximum Accuracy Configuration")
print(f"  Environment : {'Google Colab' if IS_COLAB else 'Local'}")
print(f"  Device      : {device}")
print(f"  CLEAN_DIR   : {CLEAN_DIR}")
print(f"  SAVE_DIR    : {SAVE_DIR}")
print(f"  Batch size  : {BATCH_SIZE}  |  Epochs : {NUM_EPOCHS}  |  Folds: {N_FOLDS}")
print(f"  LR          : {LR}  |  Patience: {PATIENCE}  |  Warmup: {WARMUP_EPOCHS}")
print(f"  CutMix p    : {CUTMIX_PROB}  |  Mixup α: {MIXUP_ALPHA}")
print(f"  Grad accum  : {ACCUMULATION_STEPS}  (eff. batch: {BATCH_SIZE*ACCUMULATION_STEPS})")
print(f"  Class weights: {CLASS_WEIGHTS}")
print(f"  Label smooth : {LABEL_SMOOTHING}")
print(f"{'='*65}\n")


# ══════════════════════════════════════════════════════════════════════
# BUILD DATAFRAME
# ══════════════════════════════════════════════════════════════════════
def build_df():
    if not os.path.exists(CLEAN_DIR):
        raise FileNotFoundError(f"\nCLEAN_DIR not found: {CLEAN_DIR}")

    actual_dirs = sorted([d for d in os.listdir(CLEAN_DIR)
                          if os.path.isdir(os.path.join(CLEAN_DIR, d))])
    if not actual_dirs:
        raise FileNotFoundError(f"No subfolders in {CLEAN_DIR}")

    print(f"Scanning: {CLEAN_DIR}")
    folder_label_map = {}
    for folder in actual_dirs:
        try:
            folder_label_map[folder] = int(folder.split('_')[0])
        except (ValueError, IndexError):
            print(f"  [SKIP] {folder}")

    records = []
    for folder in sorted(folder_label_map, key=folder_label_map.get):
        label   = folder_label_map[folder]
        cls_dir = os.path.join(CLEAN_DIR, folder)
        files   = [f for f in os.listdir(cls_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS]
        for f in files:
            records.append({"Image": f, "Label": label, "Folder": folder})
        bar = "█" * (len(files) // 30)
        print(f"  {folder:20s} (label={label}): {len(files):5d}  {bar}")

    df = pd.DataFrame(records)
    print(f"\n  Total: {len(df)} images across {len(folder_label_map)} classes\n")
    return df


# ══════════════════════════════════════════════════════════════════════
# MIXUP + CUTMIX
# ══════════════════════════════════════════════════════════════════════
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(x.size(0)).to(x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W, H  = size[2], size[3]
    cut_r = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_r); cut_h = int(H * cut_r)
    cx    = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H); y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=CUTMIX_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3))
    return mixed, y, y[idx], lam


# ══════════════════════════════════════════════════════════════════════
# AUGMENTATION CLASSES
# ══════════════════════════════════════════════════════════════════════
class RandomGaussianBlur:
    def __init__(self, p=0.2, radius=(0.1, 1.5)):
        self.p, self.r = p, radius
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(*self.r)))
        return img

class RandomBlackPatch:
    def __init__(self, p=0.15, max_patches=2, max_size=20):
        self.p, self.mp, self.ms = p, max_patches, max_size
    def __call__(self, img):
        if random.random() < self.p:
            img  = img.copy()
            draw = ImageDraw.Draw(img)
            w, h = img.size
            for _ in range(random.randint(1, self.mp)):
                s    = random.randint(5, self.ms)
                x, y = random.randint(0, w), random.randint(0, h)
                draw.ellipse([x-s, y-s, x+s, y+s], fill=(0, 0, 0))
        return img

class RandomBrightnessContrast:
    def __init__(self, p=0.6, brightness=(-0.25, 0.25), contrast=(0.75, 1.25)):
        self.p, self.b, self.c = p, brightness, contrast
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_brightness(img, max(0, 1 + random.uniform(*self.b)))
            img = TF.adjust_contrast(img, random.uniform(*self.c))
        return img

class RandomSaturationHue:
    def __init__(self, p=0.5, saturation=(0.8, 1.3), hue=(-0.06, 0.06)):
        self.p, self.s, self.h = p, saturation, hue
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_saturation(img, random.uniform(*self.s))
            img = TF.adjust_hue(img, random.uniform(*self.h))
        return img

class RandomSharpness:
    def __init__(self, p=0.3, sharpness=(0.8, 2.0)):
        self.p, self.s = p, sharpness
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_sharpness(img, random.uniform(*self.s))
        return img

class CLAHE_PIL:
    """CLAHE — enhances vessel and lesion contrast, critical for DR."""
    def __init__(self, p=0.6):   # increased from 0.4
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            arr     = np.array(img)
            lab     = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l       = clahe.apply(l)
            lab     = cv2.merge([l, a, b])
            arr     = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            img     = Image.fromarray(arr)
        return img

class RandomGridDistortion:
    def __init__(self, p=0.15, num_steps=5, distort_limit=0.1):
        self.p, self.ns, self.dl = p, num_steps, distort_limit
    def __call__(self, img):
        if random.random() >= self.p:
            return img
        arr    = np.array(img)
        h, w   = arr.shape[:2]
        sx, sy = max(1, w // self.ns), max(1, h // self.ns)
        src, dst = [], []
        for i in range(self.ns + 1):
            for j in range(self.ns + 1):
                px = min(j * sx, w - 1); py = min(i * sy, h - 1)
                dx = max(0, min(w-1, px + random.randint(-int(sx*self.dl), int(sx*self.dl))))
                dy = max(0, min(h-1, py + random.randint(-int(sy*self.dl), int(sy*self.dl))))
                src.append([px, py]); dst.append([dx, dy])
        M   = cv2.getPerspectiveTransform(np.float32(src[:4]), np.float32(dst[:4]))
        out = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(out)

class MixedAugmentation:
    def __init__(self, p=0.35):
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        s = random.randint(0, 3)
        if   s == 0:
            img = TF.adjust_brightness(img, random.uniform(0.6, 0.85))
            img = TF.adjust_contrast(img, random.uniform(1.1, 1.6))
        elif s == 1:
            img = TF.adjust_brightness(img, random.uniform(1.2, 1.6))
            img = TF.adjust_gamma(img, random.uniform(0.65, 0.95))
        elif s == 2:
            img = TF.adjust_saturation(img, random.uniform(0.6, 1.4))
            img = TF.adjust_hue(img, random.uniform(-0.08, 0.08))
        else:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        return img


# ══════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(25, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                            scale=(0.9, 1.1), shear=5),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
    CLAHE_PIL(p=0.6),                        # increased probability
    RandomBrightnessContrast(p=0.6),
    RandomSaturationHue(p=0.5),
    RandomSharpness(p=0.3),
    RandomGaussianBlur(p=0.2),
    RandomBlackPatch(p=0.15),
    MixedAugmentation(p=0.35),
    RandomGridDistortion(p=0.15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.04),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.01, 0.06), ratio=(0.3, 3.0), value=0),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════
class DRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        fname  = str(row["Image"])
        label  = int(row["Label"])
        folder = str(row["Folder"])
        path   = os.path.join(CLEAN_DIR, folder, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


# ══════════════════════════════════════════════════════════════════════
# WARMUP SCHEDULER
# ══════════════════════════════════════════════════════════════════════
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, T_0=10):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lrs      = [pg['lr'] for pg in optimizer.param_groups]
        self.cosine        = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=1, eta_min=1e-7
        )
        self.current_epoch = 0

    def step(self, epoch=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            scale = self.current_epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * scale
        else:
            self.cosine.step(self.current_epoch - self.warmup_epochs)

    def state_dict(self):
        return {"current_epoch": self.current_epoch,
                "cosine": self.cosine.state_dict()}

    def load_state_dict(self, d):
        self.current_epoch = d["current_epoch"]
        self.cosine.load_state_dict(d["cosine"])


# ══════════════════════════════════════════════════════════════════════
# VALIDATION WITH PROPER H-FLIP TTA
# ══════════════════════════════════════════════════════════════════════
def validate(model, val_loader, criterion):
    """
    Validation with horizontal flip TTA — averages normal + flipped
    predictions for slightly better val accuracy estimate.
    """
    model.eval()
    v_loss    = 0.0
    v_correct = 0
    v_total   = 0
    ep_preds  = []
    ep_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="  Val", leave=False):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            # Normal pass
            out_normal  = torch.softmax(model(imgs), dim=1)
            # H-flip pass
            out_flipped = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            # Average probabilities
            avg_probs   = (out_normal + out_flipped) / 2

            preds = avg_probs.argmax(dim=1)

            # Loss on normal pass only
            loss      = criterion(model(imgs), labels)
            v_loss   += loss.item() * imgs.size(0)
            v_correct += (preds == labels).sum().item()
            v_total   += labels.size(0)
            ep_preds.extend(preds.cpu().numpy())
            ep_labels.extend(labels.cpu().numpy())

    return (v_loss   / max(v_total, 1),
            v_correct / max(v_total, 1) * 100,
            ep_preds,
            ep_labels)


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════
def save_epoch_ckpt(path, fold, epoch, model, optimizer, scheduler,
                    val_acc, best_fold_f1, train_losses, val_losses,
                    train_accs, val_accs):
    torch.save({
        "fold"            : fold,
        "epoch"           : epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "scheduler_state" : scheduler.state_dict(),
        "val_acc"         : val_acc,
        "best_fold_f1"    : best_fold_f1,
        "train_losses"    : train_losses,
        "val_losses"      : val_losses,
        "train_accs"      : train_accs,
        "val_accs"        : val_accs,
    }, path)


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
def plot_overall_training_curves(all_tl, all_vl, all_ta, all_va, save_dir):
    def pad(lists):
        mx = max(len(l) for l in lists)
        return np.array([l + [float('nan')]*(mx-len(l)) for l in lists], dtype=float)

    tl = pad(all_tl); vl = pad(all_vl)
    ta = pad(all_ta); va = pad(all_va)
    ep = np.arange(1, tl.shape[1]+1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Overall Training Curves (All Folds)", fontsize=14, fontweight='bold')

    for ax, (m1, s1, m2, s2, r1, r2, title) in zip(axes, [
        (np.nanmean(tl,0), np.nanstd(tl,0),
         np.nanmean(vl,0), np.nanstd(vl,0), all_tl, all_vl, "Loss"),
        (np.nanmean(ta,0), np.nanstd(ta,0),
         np.nanmean(va,0), np.nanstd(va,0), all_ta, all_va, "Accuracy (%)"),
    ]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        for f1_, f2_ in zip(r1, r2):
            e_ = np.arange(1, len(f1_)+1)
            ax.plot(e_, f1_, color='steelblue',  alpha=0.2, linewidth=1)
            ax.plot(e_, f2_, color='darkorange', alpha=0.2, linewidth=1)
        ax.plot(ep, m1, color='steelblue',  linewidth=2, label='Train (mean)')
        ax.plot(ep, m2, color='darkorange', linewidth=2, label='Val (mean)')
        ax.fill_between(ep, m1-s1, m1+s1, color='steelblue',  alpha=0.15)
        ax.fill_between(ep, m2-s2, m2+s2, color='darkorange', alpha=0.15)
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "overall_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out}")


def plot_overall_confusion_matrix(all_labels, all_preds, save_dir):
    short = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
    cm      = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Overall Confusion Matrix (All Folds Combined)",
                 fontsize=14, fontweight='bold')
    sns.heatmap(cm,      annot=True, fmt='d',    cmap='Blues',
                xticklabels=short, yticklabels=short, ax=axes[0], linewidths=0.5)
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=short, yticklabels=short, ax=axes[1],
                linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Normalised (row %)")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    out = os.path.join(save_dir, "overall_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out}")
    print("\n  Per-class Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=short, zero_division=0))


def plot_fold_comparison(fold_results, save_dir):
    folds = [r["fold"] for r in fold_results]
    accs  = [r["acc"]  for r in fold_results]
    f1s   = [r["f1"]   for r in fold_results]
    x     = np.arange(len(folds)); w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x-w/2, accs, w, label='Val Acc (%)',  color='steelblue',  alpha=0.85)
    b2 = ax.bar(x+w/2, f1s,  w, label='Val F1 (%)',   color='darkorange', alpha=0.85)
    ax.set_title("Per-Fold Performance", fontsize=13, fontweight='bold')
    ax.set_xlabel("Fold"); ax.set_ylabel("Score (%)")
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0, 110); ax.legend(); ax.grid(axis='y', alpha=0.3)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=9)
    ax.axhline(np.mean(accs), color='steelblue',  linestyle='--', linewidth=1)
    ax.axhline(np.mean(f1s),  color='darkorange', linestyle='--', linewidth=1)
    plt.tight_layout()
    out = os.path.join(save_dir, "fold_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════
def train():
    df = build_df()
    y  = df["Label"].values

    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(CLASS_WEIGHTS).to(device),
        label_smoothing=LABEL_SMOOTHING
    )

    skf             = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results    = []
    best_overall_f1 = 0.0
    best_path       = os.path.join(SAVE_DIR, "best_model.pth")

    all_tl, all_vl, all_ta, all_va = [], [], [], []
    all_labels_combined, all_preds_combined = [], []

    _state = dict(fold=None, epoch=0, model=None, optimizer=None, scheduler=None,
                  best_fold_f1=0.0, tl=[], vl=[], ta=[], va=[])

    try:
        for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), 1):
            _state["fold"] = fold

            print(f"\n{'='*65}")
            print(f"  FOLD {fold}/{N_FOLDS}")
            print(f"{'='*65}")

            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df   = df.iloc[va_idx].reset_index(drop=True)

            t_dist = np.bincount(train_df["Label"].values, minlength=NUM_CLASSES)
            v_dist = np.bincount(val_df["Label"].values,   minlength=NUM_CLASSES)
            print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
            print(f"  Train dist: {t_dist}")
            print(f"  Val   dist: {v_dist}\n")

            sample_w = [CLASS_WEIGHTS[t] for t in train_df["Label"].values]
            sampler  = WeightedRandomSampler(
                sample_w, num_samples=int(len(sample_w) * 1.5), replacement=True
            )

            train_loader = DataLoader(
                DRDataset(train_df, train_transforms),
                batch_size=BATCH_SIZE, sampler=sampler,
                num_workers=NW, pin_memory=PIN,
                drop_last=True, persistent_workers=False,
            )
            val_loader = DataLoader(
                DRDataset(val_df, val_transforms),
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NW, pin_memory=PIN,
                persistent_workers=False,
            )

            model     = build_model(pretrained=True).to(device)
            optimizer = torch.optim.AdamW([
                {"params": model.features.denseblock2.parameters(), "lr": LR * 0.01},
                {"params": model.features.transition2.parameters(), "lr": LR * 0.01},
                {"params": model.features.denseblock3.parameters(), "lr": LR * 0.1},
                {"params": model.features.transition3.parameters(), "lr": LR * 0.1},
                {"params": model.features.denseblock4.parameters(), "lr": LR * 0.5},
                {"params": model.features.norm5.parameters(),       "lr": LR * 0.5},
                {"params": model.classifier.parameters(),           "lr": LR},
            ], weight_decay=1e-4)

            scheduler = WarmupCosineScheduler(optimizer,
                                              warmup_epochs=WARMUP_EPOCHS, T_0=10)

            # ReduceLROnPlateau on top — halves LR when val F1 stagnates
            plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5,
                patience=6, min_lr=1e-7
            )

            _state.update(model=model, optimizer=optimizer, scheduler=scheduler)

            best_fold_f1     = 0.0
            best_fold_acc    = 0.0
            best_fold_state  = None
            best_fold_preds  = []
            best_fold_labels = []
            train_losses, val_losses, train_accs, val_accs = [], [], [], []
            no_improve = 0

            for epoch in range(1, NUM_EPOCHS + 1):
                _state["epoch"] = epoch

                # ── Train ─────────────────────────────────────────────
                model.train()
                run_loss = correct = total = 0
                optimizer.zero_grad()

                for batch_idx, (imgs, labels) in enumerate(tqdm(
                        train_loader,
                        desc=f"  F{fold} Ep{epoch:02d} Train",
                        leave=False)):

                    imgs, labels = imgs.to(device), labels.to(device)

                    if random.random() < CUTMIX_PROB:
                        mixed, y_a, y_b, lam = cutmix_data(imgs, labels)
                    else:
                        mixed, y_a, y_b, lam = mixup_data(imgs, labels)

                    out  = model(mixed)
                    loss = mixup_criterion(criterion, out, y_a, y_b, lam)
                    loss = loss / ACCUMULATION_STEPS
                    loss.backward()

                    if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    run_loss += loss.item() * ACCUMULATION_STEPS * imgs.size(0)
                    correct  += (out.argmax(1) == y_a).sum().item()
                    total    += labels.size(0)

                # Flush remaining gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                train_loss = run_loss / max(total, 1)
                train_acc  = correct  / max(total, 1) * 100

                # ── Validate with H-flip TTA ──────────────────────────
                val_loss, val_acc, ep_preds, ep_labels = validate(
                    model, val_loader, criterion
                )

                # Step both schedulers
                scheduler.step(epoch)
                plateau_scheduler.step(
                    f1_score(ep_labels, ep_preds,
                             average="weighted", zero_division=0) * 100
                )

                val_f1     = f1_score(ep_labels, ep_preds,
                                      average="weighted", zero_division=0) * 100
                per_cls_f1 = f1_score(ep_labels, ep_preds,
                                      average=None, zero_division=0) * 100

                # Per-class accuracy
                per_cls_acc = []
                for c in range(NUM_CLASSES):
                    mask = np.array(ep_labels) == c
                    if mask.sum() > 0:
                        per_cls_acc.append(
                            (np.array(ep_preds)[mask] == c).sum() / mask.sum() * 100
                        )
                    else:
                        per_cls_acc.append(0.0)

                cur_lr = optimizer.param_groups[-1]["lr"]

                train_losses.append(train_loss); val_losses.append(val_loss)
                train_accs.append(train_acc);    val_accs.append(val_acc)
                _state.update(tl=train_losses, vl=val_losses,
                              ta=train_accs,   va=val_accs,
                              best_fold_f1=best_fold_f1)

                print(f"  F{fold} Ep{epoch:02d} | "
                      f"Train {train_acc:.1f}% | "
                      f"Val {val_acc:.1f}% | "
                      f"F1 {val_f1:.1f}% | "
                      f"LR {cur_lr:.2e}")
                print(f"  Per-class Acc : "
                      f"NoDR={per_cls_acc[0]:.1f}% "
                      f"Mild={per_cls_acc[1]:.1f}% "
                      f"Mod={per_cls_acc[2]:.1f}% "
                      f"Sev={per_cls_acc[3]:.1f}% "
                      f"PDR={per_cls_acc[4]:.1f}%")
                print(f"  Per-class F1  : "
                      f"NoDR={per_cls_f1[0]:.1f}% "
                      f"Mild={per_cls_f1[1]:.1f}% "
                      f"Mod={per_cls_f1[2]:.1f}% "
                      f"Sev={per_cls_f1[3]:.1f}% "
                      f"PDR={per_cls_f1[4]:.1f}%")

                save_epoch_ckpt(
                    os.path.join(SAVE_DIR, f"fold_{fold}_last.pth"),
                    fold, epoch, model, optimizer, scheduler,
                    val_acc, best_fold_f1,
                    train_losses, val_losses, train_accs, val_accs,
                )

                if val_f1 > best_fold_f1:
                    prev_f1          = best_fold_f1
                    best_fold_f1     = val_f1
                    best_fold_acc    = val_acc
                    best_fold_state  = copy.deepcopy(model.state_dict())
                    best_fold_preds  = ep_preds
                    best_fold_labels = ep_labels
                    no_improve       = 0
                    print(f"  ✓ New best F1: {val_f1:.2f}%  (↑ from {prev_f1:.2f}%)")

                    torch.save({"fold": fold, "epoch": epoch,
                                "model_state_dict": best_fold_state,
                                "val_acc": val_acc, "val_f1": best_fold_f1},
                               os.path.join(SAVE_DIR, f"fold_{fold}_best.pth"))

                    if best_fold_f1 > best_overall_f1:
                        best_overall_f1 = best_fold_f1
                        torch.save({"fold": fold, "epoch": epoch,
                                    "model_state_dict": best_fold_state,
                                    "val_acc": val_acc, "val_f1": best_fold_f1},
                                   best_path)
                        print(f"  ★ New overall best F1: {best_fold_f1:.2f}%"
                              f" → best_model.pth")
                else:
                    no_improve += 1
                    print(f"  No improvement ({no_improve}/{PATIENCE})")
                    if no_improve >= PATIENCE:
                        print(f"  Early stopping at epoch {epoch}.")
                        break

            # ── End of fold ───────────────────────────────────────────
            if best_fold_state is None:
                best_fold_state  = copy.deepcopy(model.state_dict())
                best_fold_preds  = ep_preds
                best_fold_labels = ep_labels

            fold_results.append({
                "fold": fold, "acc": best_fold_acc, "f1": best_fold_f1,
                "preds": best_fold_preds, "labels": best_fold_labels
            })

            all_tl.append(train_losses); all_vl.append(val_losses)
            all_ta.append(train_accs);   all_va.append(val_accs)
            all_labels_combined.extend(best_fold_labels)
            all_preds_combined.extend(best_fold_preds)

            fold_save = os.path.join(SAVE_DIR, f"fold_{fold}")
            os.makedirs(fold_save, exist_ok=True)
            plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                                 save_dir=fold_save)
            print(f"\n  Fold {fold} → Acc: {best_fold_acc:.2f}%  F1: {best_fold_f1:.2f}%")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving emergency checkpoint…")
        f = _state["fold"]
        if f and _state["model"]:
            epath = os.path.join(SAVE_DIR,
                                 f"fold_{f}_INTERRUPTED_ep{_state['epoch']}.pth")
            save_epoch_ckpt(
                epath, f, _state["epoch"],
                _state["model"], _state["optimizer"], _state["scheduler"],
                0.0, _state["best_fold_f1"],
                _state["tl"], _state["vl"], _state["ta"], _state["va"],
            )
            print(f"Saved to: {epath}")
        raise

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════
    if not fold_results:
        return

    print(f"\n{'='*65}")
    print("  CROSS VALIDATION COMPLETE")
    print(f"{'='*65}")
    accs = [r["acc"] for r in fold_results]
    f1s  = [r["f1"]  for r in fold_results]
    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['acc']:.2f}%  F1={r['f1']:.2f}%")
    print(f"\n  Mean Acc : {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print(f"  Mean F1  : {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
    print(f"  Best model → {best_path}  (by weighted F1)")

    print(f"\n  Generating plots …")
    plot_overall_training_curves(all_tl, all_vl, all_ta, all_va, save_dir=SAVE_DIR)
    plot_overall_confusion_matrix(all_labels_combined, all_preds_combined,
                                  save_dir=SAVE_DIR)
    plot_fold_comparison(fold_results, save_dir=SAVE_DIR)

    best = fold_results[int(np.argmax(f1s))]
    print(f"\n  Best fold: {best['fold']}  Acc={best['acc']:.2f}%  F1={best['f1']:.2f}%")
    print_metrics(best["labels"], best["preds"])
    plot_confusion_matrix(best["labels"], best["preds"], save_dir=SAVE_DIR)


if __name__ == "__main__":
    train()