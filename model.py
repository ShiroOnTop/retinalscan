import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 5

def build_model(pretrained=True):
    """
    DenseNet121 — tuned for pushing all classes above 80%.
    - Unfreeze from denseblock2 onwards (stable config)
    - Wider classifier with more dropout for Moderate/PDR separation
    """
    model = models.densenet121(
        weights="IMAGENET1K_V1" if pretrained else None
    )

    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze denseblock2 onwards
    for name in ['denseblock2', 'transition2', 'denseblock3',
                 'transition3', 'denseblock4', 'norm5']:
        for param in getattr(model.features, name).parameters():
            param.requires_grad = True

    # Wider classifier for better Moderate/PDR separation
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.35),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.15),
        nn.Linear(256, NUM_CLASSES)
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


if __name__ == "__main__":
    model = build_model(pretrained=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable : {trainable:,}  ({trainable/total*100:.1f}%)")
    print(f"Frozen    : {total-trainable:,}  ({(total-trainable)/total*100:.1f}%)")
    print(f"Total     : {total:,}")
    dummy  = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"Output    : {output.shape}")