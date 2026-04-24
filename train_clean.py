"""
train_clean.py — Training from teacher_clean/ folder only.

NO CSV required. NO raw images required.
Scans CLEAN_DIR class subfolders directly to build the dataset.

Works on both local Windows and Google Colab.

Usage:
    python train_clean.py
    !python train_clean.py   (Colab)
"""
import os
import cv2
import copy
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

from model import build_model
from utils import plot_confusion_matrix, print_metrics, plot_training_curves

# ══════════════════════════════════════════════════════════════════════
# ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════════════
try:
    import google.colab  # noqa: F401
    IS_COLAB = True
    from google.colab import drive as _gdrive
    if not os.path.exists('/content/drive/MyDrive'):
        _gdrive.mount('/content/drive')
    print("Google Colab detected — Drive mounted.")
except ImportError:
    IS_COLAB = False

# ══════════════════════════════════════════════════════════════════════
# CONFIG  — only two paths needed
# ══════════════════════════════════════════════════════════════════════
if IS_COLAB:
    CLEAN_DIR = "/content/drive/MyDrive/project/dataset/teacher_clean"
    SAVE_DIR  = "/content/drive/MyDrive/project/saved_model"
else:
    CLEAN_DIR = r"C:\Users\xiaon\Downloads\dr ai\project\dataset\teacher_clean"
    SAVE_DIR  = "saved_model"

NUM_CLASSES  = 5
IMG_SIZE     = 224
BATCH_SIZE   = 16 if IS_COLAB else 8
NUM_EPOCHS   = 25
LR           = 1e-4
N_FOLDS      = 5
SEED         = 42
PATIENCE     = 6

NW  = 2 if IS_COLAB else 0
PIN = torch.cuda.is_available()

CLASS_NAMES = ["0_No_DR", "1_Mild", "2_Moderate", "3_Severe", "4_PDR"]
IMG_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\n{'='*55}")
print("  RetinaScan AI — Train from Clean Folder")
print(f"  Stratified {N_FOLDS}-Fold Cross Validation")
print(f"  Environment: {'Google Colab' if IS_COLAB else 'Local'}")
print(f"  Device     : {device}")
print(f"  Epochs/fold: {NUM_EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LR         : {LR}")
print(f"  num_workers: {NW}  |  pin_memory: {PIN}")
print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════════════
# BUILD DATASET FROM FOLDER  (no CSV needed)
# ══════════════════════════════════════════════════════════════════════

def build_df_from_clean_dir():
    """Scan teacher_clean/<class>/ folders and return a DataFrame."""
    records = []
    print(f"Scanning: {CLEAN_DIR}\n")

    for label_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(CLEAN_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [WARN] Missing folder: {cls_dir} — skipping.")
            continue

        fnames = [f for f in os.listdir(cls_dir)
                  if os.path.splitext(f)[1].lower() in IMG_EXTS]

        for fname in fnames:
            records.append({"Image": fname, "Label": label_idx})

        bar = "█" * (len(fnames) // 30)
        print(f"  {cls:20s}: {len(fnames):5d}  {bar}")

    df = pd.DataFrame(records)
    print(f"\n  Total: {len(df)} images\n")
    return df


# ══════════════════════════════════════════════════════════════════════
# CUSTOM AUGMENTATION CLASSES
# ══════════════════════════════════════════════════════════════════════

class RandomGaussianBlur:
    def __init__(self, p=0.25, radius_range=(0.1, 1.8)):
        self.p, self.r = p, radius_range
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(*self.r)))
        return img


class RandomBlackPatch:
    def __init__(self, p=0.2, max_patches=3, max_size=25):
        self.p, self.mp, self.ms = p, max_patches, max_size
    def __call__(self, img):
        if random.random() < self.p:
            img = img.copy()
            draw = ImageDraw.Draw(img)
            w, h = img.size
            for _ in range(random.randint(1, self.mp)):
                s = random.randint(5, self.ms)
                x, y = random.randint(0, w), random.randint(0, h)
                draw.ellipse([x-s, y-s, x+s, y+s], fill=(0, 0, 0))
        return img


class RandomGridDistortion:
    def __init__(self, p=0.15, num_steps=5, distort_limit=0.1):
        self.p, self.ns, self.dl = p, num_steps, distort_limit
    def __call__(self, img):
        if random.random() >= self.p:
            return img
        arr = np.array(img)
        h, w = arr.shape[:2]
        sx, sy = max(1, w // self.ns), max(1, h // self.ns)
        src, dst = [], []
        for i in range(self.ns + 1):
            for j in range(self.ns + 1):
                sx_ = min(j * sx, w - 1)
                sy_ = min(i * sy, h - 1)
                dx = sx_ + random.randint(-int(sx * self.dl), int(sx * self.dl))
                dy = sy_ + random.randint(-int(sy * self.dl), int(sy * self.dl))
                src.append([sx_, sy_])
                dst.append([max(0, min(w-1, dx)), max(0, min(h-1, dy))])
        M = cv2.getPerspectiveTransform(np.float32(src[:4]), np.float32(dst[:4]))
        out = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return Image.fromarray(out)


class RandomBrightnessContrast:
    def __init__(self, p=0.6, brightness=(-0.3, 0.3), contrast=(0.7, 1.3)):
        self.p, self.b, self.c = p, brightness, contrast
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_brightness(img, max(0, 1 + random.uniform(*self.b)))
            img = TF.adjust_contrast(img, random.uniform(*self.c))
        return img


class RandomSaturationHue:
    def __init__(self, p=0.5, saturation=(0.7, 1.4), hue=(-0.08, 0.08)):
        self.p, self.s, self.h = p, saturation, hue
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_saturation(img, random.uniform(*self.s))
            img = TF.adjust_hue(img, random.uniform(*self.h))
        return img


class RandomSharpness:
    def __init__(self, p=0.3, sharpness=(0.5, 2.5)):
        self.p, self.s = p, sharpness
    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_sharpness(img, random.uniform(*self.s))
        return img


class MixedAugmentation:
    def __init__(self, p=0.4):
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        s = random.randint(0, 3)
        if s == 0:
            img = TF.adjust_brightness(img, random.uniform(0.5, 0.8))
            img = TF.adjust_contrast(img, random.uniform(1.2, 1.8))
        elif s == 1:
            img = TF.adjust_brightness(img, random.uniform(1.3, 1.7))
            img = TF.adjust_gamma(img, random.uniform(0.6, 0.9))
        elif s == 2:
            img = TF.adjust_saturation(img, random.uniform(0.5, 1.5))
            img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
        else:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        return img


# ══════════════════════════════════════════════════════════════════════
# TRANSFORMS
# ══════════════════════════════════════════════════════════════════════

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(25, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomAffine(degrees=15, translate=(0.12, 0.12), scale=(0.88, 1.12), shear=5),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
    RandomBrightnessContrast(p=0.6),
    RandomSaturationHue(p=0.5),
    RandomSharpness(p=0.3),
    RandomGaussianBlur(p=0.25),
    RandomBlackPatch(p=0.2),
    MixedAugmentation(p=0.4),
    RandomGridDistortion(p=0.15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.01, 0.08), ratio=(0.3, 3.0), value=0),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════

class DRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fname = str(row["Image"])
        label = int(row["Label"])
        cls   = CLASS_NAMES[label]
        path  = os.path.join(CLEAN_DIR, cls, fname)

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label


# ══════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════

def save_epoch_ckpt(path, fold, epoch, model, optimizer, scheduler,
                    val_acc, best_fold_acc, train_losses, val_losses,
                    train_accs, val_accs):
    torch.save({
        "fold"             : fold,
        "epoch"            : epoch,
        "model_state_dict" : model.state_dict(),
        "optimizer_state"  : optimizer.state_dict(),
        "scheduler_state"  : scheduler.state_dict(),
        "val_acc"          : val_acc,
        "best_fold_acc"    : best_fold_acc,
        "train_losses"     : train_losses,
        "val_losses"       : val_losses,
        "train_accs"       : train_accs,
        "val_accs"         : val_accs,
    }, path)


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def train():
    # ── Build dataset from folder ──────────────────────────────────────
    df = build_df_from_clean_dir()

    if len(df) == 0:
        print("ERROR: No images found in CLEAN_DIR. Check your path:")
        print(f"  {CLEAN_DIR}")
        return

    y = df["Label"].values

    loss_weights = torch.FloatTensor([1.0, 2.0, 1.5, 3.0, 2.5]).to(device)
    criterion    = nn.CrossEntropyLoss(weight=loss_weights)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results     = []
    best_overall_acc = 0.0
    best_model_path  = os.path.join(SAVE_DIR, "best_model.pth")

    current_fold = None
    model = optimizer = scheduler = None
    epoch = 0
    train_losses = val_losses = train_accs = val_accs = []
    best_fold_acc = 0.0

    try:
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), 1):
            current_fold = fold

            print(f"\n{'='*55}")
            print(f"  FOLD {fold}/{N_FOLDS}")
            print(f"{'='*55}")

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df   = df.iloc[val_idx].reset_index(drop=True)

            t_dist = np.bincount(train_df["Label"].values, minlength=NUM_CLASSES)
            v_dist = np.bincount(val_df["Label"].values,   minlength=NUM_CLASSES)
            print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
            print(f"  Train dist: {t_dist}")
            print(f"  Val dist  : {v_dist}\n")

            train_ds = DRDataset(train_df, train_transforms)
            val_ds   = DRDataset(val_df,   val_transforms)

            # WeightedRandomSampler
            oversample_w = np.array([1.0, 2.0, 1.5, 4.0, 3.0])
            sample_w     = [oversample_w[t] for t in train_df["Label"].values]
            sampler      = WeightedRandomSampler(
                sample_w,
                num_samples=int(len(sample_w) * 1.5),
                replacement=True,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=NW,
                pin_memory=PIN,
                drop_last=True,         # prevents BatchNorm crash on last batch of size 1
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NW,
                pin_memory=PIN,
            )

            model = build_model(pretrained=True).to(device)

            optimizer = torch.optim.AdamW([
                {"params": model.features.denseblock3.parameters(), "lr": LR * 0.1},
                {"params": model.features.transition3.parameters(), "lr": LR * 0.1},
                {"params": model.features.denseblock4.parameters(), "lr": LR * 0.5},
                {"params": model.features.norm5.parameters(),       "lr": LR * 0.5},
                {"params": model.classifier.parameters(),           "lr": LR},
            ], weight_decay=1e-4)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
            )

            best_fold_acc   = 0.0
            best_fold_state = None
            best_fold_preds = []
            best_fold_labels= []
            train_losses    = []
            val_losses      = []
            train_accs      = []
            val_accs        = []
            no_improve      = 0

            for epoch in range(1, NUM_EPOCHS + 1):

                # ── Train ───────────────────────────────────────────
                model.train()
                run_loss, correct, total = 0.0, 0, 0

                for imgs, labels in tqdm(
                    train_loader,
                    desc=f"  F{fold} Ep{epoch:02d} [Train]",
                    leave=False,
                ):
                    imgs, labels = imgs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    out  = model(imgs)
                    loss = criterion(out, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    run_loss += loss.item() * imgs.size(0)
                    _, preds  = torch.max(out, 1)
                    correct  += (preds == labels).sum().item()
                    total    += labels.size(0)

                train_loss = run_loss / max(total, 1)
                train_acc  = correct  / max(total, 1) * 100

                # ── Validate ─────────────────────────────────────────
                model.eval()
                v_loss, v_correct, v_total = 0.0, 0, 0
                all_preds, all_labels = [], []

                with torch.no_grad():
                    for imgs, labels in tqdm(
                        val_loader,
                        desc=f"  F{fold} Ep{epoch:02d} [Val]  ",
                        leave=False,
                    ):
                        imgs, labels = imgs.to(device), labels.to(device)
                        out  = model(imgs)
                        loss = criterion(out, labels)

                        v_loss    += loss.item() * imgs.size(0)
                        _, preds   = torch.max(out, 1)
                        v_correct += (preds == labels).sum().item()
                        v_total   += labels.size(0)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                val_loss = v_loss    / max(v_total, 1)
                val_acc  = v_correct / max(v_total, 1) * 100
                val_f1   = f1_score(all_labels, all_preds,
                                    average="weighted", zero_division=0) * 100

                scheduler.step()

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)

                cur_lr = optimizer.param_groups[-1]["lr"]
                print(
                    f"  F{fold} Ep{epoch:02d} | "
                    f"Train: {train_acc:.1f}% | "
                    f"Val: {val_acc:.1f}% | "
                    f"F1: {val_f1:.1f}% | "
                    f"LR: {cur_lr:.2e}"
                )

                # ── Save every-epoch checkpoint ──────────────────────
                last_path = os.path.join(SAVE_DIR, f"fold_{fold}_last.pth")
                save_epoch_ckpt(
                    last_path, fold, epoch, model, optimizer, scheduler,
                    val_acc, best_fold_acc,
                    train_losses, val_losses, train_accs, val_accs,
                )

                # ── Save best ────────────────────────────────────────
                if val_acc > best_fold_acc:
                    best_fold_acc    = val_acc
                    best_fold_state  = copy.deepcopy(model.state_dict())
                    best_fold_preds  = all_preds
                    best_fold_labels = all_labels
                    no_improve       = 0

                    print(f"  ✓ New best: {val_acc:.2f}%")

                    # Save fold best
                    torch.save({
                        "fold": fold, "epoch": epoch,
                        "model_state_dict": best_fold_state,
                        "val_acc": best_fold_acc,
                    }, os.path.join(SAVE_DIR, f"fold_{fold}_best.pth"))

                    # Save overall best
                    if best_fold_acc > best_overall_acc:
                        best_overall_acc = best_fold_acc
                        torch.save({
                            "fold": fold, "epoch": epoch,
                            "model_state_dict": best_fold_state,
                            "val_acc": best_fold_acc,
                        }, best_model_path)
                        print(f"  ★ New overall best: {best_fold_acc:.2f}% → best_model.pth")
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE:
                        print(f"  Early stopping at epoch {epoch}")
                        break

            # ── After fold ───────────────────────────────────────────
            if best_fold_state is None:
                best_fold_state  = copy.deepcopy(model.state_dict())
                best_fold_preds  = all_preds
                best_fold_labels = all_labels

            fold_f1 = f1_score(
                best_fold_labels, best_fold_preds,
                average="weighted", zero_division=0,
            ) * 100

            fold_results.append({
                "fold": fold, "val_acc": best_fold_acc, "val_f1": fold_f1,
                "preds": best_fold_preds, "labels": best_fold_labels,
            })

            print(f"\n  Fold {fold} complete → Acc: {best_fold_acc:.2f}%  F1: {fold_f1:.2f}%")

            plot_training_curves(
                train_losses, val_losses, train_accs, val_accs,
                save_dir=os.path.join(SAVE_DIR, f"fold_{fold}"),
            )

    except KeyboardInterrupt:
        print("\nInterrupted! Saving emergency checkpoint…")
        if model is not None and current_fold is not None:
            epath = os.path.join(SAVE_DIR, f"fold_{current_fold}_INTERRUPTED.pth")
            save_epoch_ckpt(
                epath, current_fold, epoch, model, optimizer, scheduler,
                0.0, best_fold_acc,
                train_losses, val_losses, train_accs, val_accs,
            )
            print(f"Saved to: {epath}")
        raise

    # ── Final summary ────────────────────────────────────────────────
    if not fold_results:
        print("No fold results available.")
        return

    print(f"\n{'='*55}")
    print("  CROSS VALIDATION COMPLETE")
    print(f"{'='*55}")

    accs = [r["val_acc"] for r in fold_results]
    f1s  = [r["val_f1"]  for r in fold_results]

    for r in fold_results:
        print(f"  Fold {r['fold']}: Acc={r['val_acc']:.2f}%  F1={r['val_f1']:.2f}%")

    print(f"\n  Mean Accuracy : {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
    print(f"  Mean F1 Score : {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
    print(f"  Best model    : {best_model_path}")

    best_idx  = int(np.argmax(accs))
    best_data = fold_results[best_idx]
    print(f"\n  Metrics from best fold ({best_data['fold']}):")
    print_metrics(best_data["labels"], best_data["preds"])
    plot_confusion_matrix(best_data["labels"], best_data["preds"], save_dir=SAVE_DIR)


if __name__ == "__main__":
    train()
