import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)
import seaborn as sns
import torch

CLASS_NAMES = ['0_No_DR', '1_Mild', '2_Moderate', '3_Severe', '4_PDR']

# ── 1. Plot training curves ──────────────────────────────────────────
def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_dir='saved_model'):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses,   'r-o', label='Val Loss')
    ax1.set_title('Loss per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, val_accs,   'r-o', label='Val Accuracy')
    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    print(f"[Saved] training_curves.png")


# ── 2. Plot confusion matrix ─────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_dir='saved_model'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"[Saved] confusion_matrix.png")


# ── 3. Print all metrics ─────────────────────────────────────────────
def print_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred) * 100
    f1   = f1_score(y_true, y_pred, average='weighted') * 100
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    print("\n" + "="*45)
    print(f"  Accuracy  : {acc:.2f}%")
    print(f"  F1 Score  : {f1:.2f}%")
    print(f"  Precision : {prec:.2f}%")
    print(f"  Recall    : {rec:.2f}%")
    print("="*45)
    print("\nPer-Class Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return acc, f1, prec, rec


# ── 4. Save model checkpoint ─────────────────────────────────────────
def save_checkpoint(model, epoch, val_acc, save_dir='saved_model'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'best_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc
    }, path)
    print(f"[Saved] best_model.pth  (epoch {epoch}, val_acc {val_acc:.2f}%)")


# ── 5. Load model checkpoint ─────────────────────────────────────────
def load_checkpoint(model, path='saved_model/best_model.pth'):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Loaded] {path}  (epoch {checkpoint['epoch']}, val_acc {checkpoint['val_acc']:.2f}%)")
    return model