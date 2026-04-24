import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import build_model
from utils import (plot_training_curves, plot_confusion_matrix,
                   print_metrics, save_checkpoint)

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
DATA_DIR    = 'dataset/train'
SAVE_DIR    = 'saved_model'
NUM_CLASSES = 5
IMG_SIZE    = 224
BATCH_SIZE  = 8
NUM_EPOCHS  = 30
LR          = 5e-5        # lower LR — prevents overfitting
VAL_SPLIT   = 0.2
PATIENCE    = 7           # early stopping patience
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*55}")
print(f"  Device     : {device}")
print(f"  Epochs     : {NUM_EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LR         : {LR}")
print(f"{'='*55}\n")

# ══════════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════════════
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(25),
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.4,
        hue=0.15
    ),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.15, 0.15),
        scale=(0.85, 1.15)
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════
def get_dataloaders():
    full_dataset = datasets.ImageFolder(
        root=DATA_DIR,
        transform=train_transforms
    )

    total      = len(full_dataset)
    val_size   = int(total * VAL_SPLIT)
    train_size = total - val_size

    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Apply val transforms
    val_ds.dataset.transform = val_transforms

    # Class distribution
    targets      = [full_dataset.targets[i] for i in train_ds.indices]
    class_counts = np.bincount(targets)

    print("Dataset loaded:")
    print(f"  Total  : {total}")
    print(f"  Train  : {train_size}")
    print(f"  Val    : {val_size}")
    print(f"  Classes: {full_dataset.classes}")
    print(f"\nClass distribution (train):")
    class_names = ['0_No_DR','1_Mild','2_Moderate','3_Severe','4_PDR']
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        bar = '█' * (count // 30)
        print(f"  {name:15s}: {count:5d}  {bar}")

    # ── Weighted sampler — boost rare classes ──────────────────────
    oversample_weights = np.array([
        0.5,   # 0_No_DR    — very common, reduce
        2.5,   # 1_Mild     — rare
        1.0,   # 2_Moderate — moderate
        6.0,   # 3_Severe   — very rare, heavy boost
        5.0,   # 4_PDR      — very rare, heavy boost
    ])
    sample_weights = [oversample_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=int(len(sample_weights) * 1.5),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, class_counts

# ══════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════
def train():
    train_loader, val_loader, class_counts = get_dataloaders()

    model = build_model(pretrained=True).to(device)

    # ── Weighted loss — penalise errors on rare classes more ──────────
    loss_weights = torch.FloatTensor([
        0.5,   # 0_No_DR
        2.5,   # 1_Mild
        1.0,   # 2_Moderate
        6.0,   # 3_Severe  ← heavy penalty for missing these
        5.0,   # 4_PDR     ← heavy penalty for missing these
    ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # ── Differential LR — backbone learns slower than head ───────────
    optimizer = torch.optim.AdamW([
        {'params': model.features.denseblock3.parameters(), 'lr': LR * 0.1},
        {'params': model.features.transition3.parameters(), 'lr': LR * 0.1},
        {'params': model.features.denseblock4.parameters(), 'lr': LR * 0.5},
        {'params': model.features.norm5.parameters(),       'lr': LR * 0.5},
        {'params': model.classifier.parameters(),           'lr': LR},
    ], weight_decay=1e-4)

    # ── Cosine annealing — smooth LR decay ───────────────────────────
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-7
    )

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc  = 0.0
    no_improve    = 0
    best_preds    = []
    best_labels   = []

    print(f"\n{'='*55}")
    print("  Starting training...")
    print(f"{'='*55}\n")

    for epoch in range(1, NUM_EPOCHS + 1):

        # ── Train phase ───────────────────────────────────────────────
        model.train()
        run_loss = 0.0
        correct  = 0
        total    = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Train]",
            leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            run_loss += loss.item() * images.size(0)
            _, preds  = torch.max(outputs, 1)
            correct  += (preds == labels).sum().item()
            total    += labels.size(0)

        train_loss = run_loss / total
        train_acc  = correct / total * 100

        # ── Val phase ─────────────────────────────────────────────────
        model.eval()
        v_loss   = 0.0
        v_correct= 0
        v_total  = 0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Val]  ",
                leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                v_loss   += loss.item() * images.size(0)
                _, preds  = torch.max(outputs, 1)
                v_correct+= (preds == labels).sum().item()
                v_total  += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total * 100

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Current LR
        cur_lr = optimizer.param_groups[-1]['lr']

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.1f}% | "
              f"LR: {cur_lr:.2e}")

        # ── Save best ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            best_preds   = all_preds
            best_labels  = all_labels
            save_checkpoint(model, epoch, val_acc, SAVE_DIR)
            print(f"  ✅ New best: {val_acc:.2f}%")
        else:
            no_improve += 1
            print(f"  ⏳ No improvement {no_improve}/{PATIENCE}")

        # ── Early stopping ────────────────────────────────────────────
        if no_improve >= PATIENCE:
            print(f"\n⛔ Early stopping triggered at epoch {epoch}")
            print(f"   Best val accuracy: {best_val_acc:.2f}%")
            break

    # ── Final results ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*55}")

    print_metrics(best_labels, best_preds)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, SAVE_DIR)
    plot_confusion_matrix(best_labels, best_preds, SAVE_DIR)


if __name__ == '__main__':
    train()