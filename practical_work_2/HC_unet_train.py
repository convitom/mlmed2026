import torch
from torch.utils.data import Dataset
import cv2, pandas as pd, numpy as np
import cv2
import numpy as np
import os

EDGE_DIR = "training_set/annos_edge"
MASK_DIR = "training_set/masks_filled"
IMG_SIZE = 256
os.makedirs(MASK_DIR, exist_ok=True)

for fname in os.listdir(EDGE_DIR):
    edge = cv2.imread(os.path.join(EDGE_DIR, fname), 0)
    _, edge = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        continue

    cnt = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(edge)
    cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)

    cv2.imwrite(os.path.join(MASK_DIR, fname), filled)

class FetalDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(f"training_set/images/{row['filename']}", 0)
        mask = cv2.imread(f"training_set/masks_filled/{row['filename']}", 0)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        img = img / 255.0
        mask = (mask > 0).astype(np.float32)

        img = np.stack([img]*3, axis=0)   # (3,H,W)
        mask = mask[None,:,:]             # (1,H,W)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            row["pixel size(mm)"],
            row["head circumference (mm)"]
        )

import segmentation_models_pytorch as smp
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = smp.losses.SoftBCEWithLogitsLoss()
def loss_fn(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

from torch.utils.data import DataLoader

dataset = FetalDataset("training_set/training_set_pixel_size_and_HC.csv")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model.train()
for epoch in range(20):
    print("Epoch loop entered", epoch)
    total_loss = 0
    for imgs, masks, _, _ in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

import math, cv2

def hc_from_pred(pred_mask, pixel_size):
    mask = (pred_mask > 0).astype(np.uint8)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    (_, _), (maj, min_), _ = cv2.fitEllipse(cnt)
    a, b = maj/2, min_/2

    hc_px = math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))
    return hc_px * pixel_size

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for img, _, px, hc_gt in dataset:
        pred = model(img.unsqueeze(0).to(DEVICE))
        pred = torch.sigmoid(pred)[0,0].cpu().numpy()

        hc = hc_from_pred(pred, px)
        if hc is not None:
            y_true.append(hc_gt)
            y_pred.append(hc)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("MAE  (mm):", mean_absolute_error(y_true, y_pred))
print("RMSE (mm):", mean_squared_error(y_true, y_pred, squared=False))
print("R²       :", r2_score(y_true, y_pred))
