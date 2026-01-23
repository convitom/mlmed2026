import os
import cv2
import math
import torch
import pandas as pd
import numpy as np
import segmentation_models_pytorch as smp

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "test_set/images"
CSV_PATH  = "test_set/test_set_pixel_size.csv"
OUT_DIR   = "test_set/ellipse_outputs"
OUT_CSV   = "test_set/hc_predictions.csv"
MODEL_PATH = r"mlmed2026\practical_work_2\HC_unet_model.pth"   
THRESHOLD = 0.5
os.makedirs(OUT_DIR, exist_ok=True)


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def preprocess_image(img_path):
    img = cv2.imread(img_path, 0)
    orig = img.copy()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.stack([img]*3, axis=0)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img, orig

def hc_from_mask(mask, pixel_size):
    mask = (mask > THRESHOLD).astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, None

    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None, None

    (cx, cy), (maj, min_), angle = cv2.fitEllipse(cnt)
    a, b = maj/2, min_/2

    hc_px = math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))
    hc_mm = hc_px * pixel_size

    return hc_mm, ((cx, cy), (maj, min_), angle)


df = pd.read_csv(CSV_PATH)
results = []

with torch.no_grad():
    for _, row in df.iterrows():
        fname = row["filename"]
        px_size = row["pixel size(mm)"]
        hc_gt = row["head circumference (mm)"]

        img_path = os.path.join(IMAGE_DIR, fname)
        img_tensor, orig_img = preprocess_image(img_path)

        pred = model(img_tensor.to(DEVICE))
        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

        pred = cv2.resize(pred, orig_img.shape[::-1])
        hc, ellipse = hc_from_mask(pred, px_size)

        vis = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        if ellipse is not None:
            cv2.ellipse(vis, ellipse, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(OUT_DIR, fname), vis)

        results.append({
            "filename": fname,
            "HC_predicted_mm": hc,
            "HC_ground_truth_mm": hc_gt
        })


pd.DataFrame(results).to_csv(OUT_CSV, index=False)

print("✅ DONE")
print(f"📁 Ellipse images: {OUT_DIR}")
print(f"📄 HC CSV: {OUT_CSV}")
