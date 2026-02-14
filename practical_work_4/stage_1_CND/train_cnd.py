# -*- coding: utf-8 -*-
"""
Training script for Candidate Nodule Detection Network (Stage 1)
Improved version:
- AMP enabled
- Save last & best model
- Resume training
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from config_cnd import config
from data_loader_cnd import LunaDataset, collate_fn
from loss_cnd import DetectionLoss

import sys
sys.path.append('/content')
from GCSAM_CND import MyModel


# =========================
# Train One Epoch (AMP)
# =========================
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device):
    model.train()

    epoch_loss = 0
    total_pos = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in pbar:
        images = batch['images'].to(device)
        bboxes = batch['bboxes']

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss_dict = criterion(outputs, bboxes)
            loss = loss_dict['total_loss']

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        total_pos += loss_dict['num_pos']

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "pos": loss_dict['num_pos']
        })

    return epoch_loss / len(train_loader), total_pos


# =========================
# Validation (AMP)
# =========================
def validate(model, val_loader, criterion, epoch, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val {epoch+1}")

        for batch in pbar:
            images_list = []
            bboxes_list = []

            for item in batch:
                if len(item['images']) > 0:
                    images_list.append(item['images'][0])
                    bboxes_list.append(item['bboxes'][0])

            if len(images_list) == 0:
                continue

            images = torch.stack(images_list).to(device)

            with autocast():
                outputs = model(images)
                loss_dict = criterion(outputs, bboxes_list)

            val_loss += loss_dict['total_loss'].item()

    return val_loss / len(val_loader)


# =========================
# Save Checkpoint
# =========================
def save_checkpoint(model, optimizer, scaler, epoch, save_path, is_best=False):
    os.makedirs(save_path, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }

    # Save last (overwrite)
    last_path = os.path.join(save_path, "last_model.pth")
    torch.save(checkpoint, last_path)

    # Save best
    if is_best:
        best_path = os.path.join(save_path, "best_model.pth")
        torch.save(checkpoint, best_path)


# =========================
# Main
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_path = config.get("save_path", "/content/checkpoints")
    os.makedirs(save_path, exist_ok=True)

    # Dataset
    train_dataset = LunaDataset(
        data_dir="/content/data",
        annotations_file="/content/data/annotations.csv",
        subset_ids=config['train_split'],
        config=config,
        phase="train"
    )

    val_dataset = LunaDataset(
        data_dir="/content/data",
        annotations_file="/content/data/annotations.csv",
        subset_ids=config['val_split'],
        config=config,
        phase="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Model
    model = MyModel().to(device)
    criterion = DetectionLoss(config)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr_stage1'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    scaler = GradScaler()

    # Resume from checkpoint
    last_path = os.path.join(save_path, "last_model.pth")
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(last_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(last_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Training loop
    for epoch in range(start_epoch, config['epoch']):
        print(f"\nEpoch {epoch+1}/{config['epoch']}")

        train_loss, total_pos = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, device
        )

        print(f"Train Loss: {train_loss:.4f} | Pos anchors: {total_pos}")

        # Validate every epoch
        val_loss = validate(model, val_loader, criterion, epoch, device)
        print(f"Val Loss: {val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print("New best model!")

        save_checkpoint(model, optimizer, scaler, epoch, save_path, is_best)

    print("Training completed!")


if __name__ == "__main__":
    main()
