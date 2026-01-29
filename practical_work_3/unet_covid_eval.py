import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from unet_covid_train import CovidDataset

DO_METRICS = 1 # Công tắc :))
SAVE_PRED_MASK = 0
SAVE_OVERLAY = 0
PR_CURVE = 0
MODEL = "unet_covid_2.pth"
IMG_SIZE = 256
PRED_MASK_DIR = "pred_infection_masks_2_(th=0.25)"
OVERLAY_DIR = "overlay_2_(th=0.25)"
threshold = 0.25

test_roots = [
    #r"D:\USTH\MLmed\anasmohammedtahir\covidqu\versions\7\Infection_Segmentation_Data\Infection_Segmentation_Data\Test\COVID-19",
    r"D:\USTH\MLmed\anasmohammedtahir\covidqu\versions\7\Infection_Segmentation_Data\Infection_Segmentation_Data\Test\Non-COVID",
    r"D:\USTH\MLmed\anasmohammedtahir\covidqu\versions\7\Infection_Segmentation_Data\Infection_Segmentation_Data\Test\Normal",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

def segmentation_metrics(pred, target, eps=1e-7):
    
    # pred, target: (B,1,H,W), binary {0,1}
    
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    TP = (pred * target).sum(dim=1)
    TN = ((1 - pred) * (1 - target)).sum(dim=1)
    FP = (pred * (1 - target)).sum(dim=1)
    FN = ((1 - pred) * target).sum(dim=1)

    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)

    return {
        "accuracy": accuracy.mean(),
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": f1.mean(),
        "dice": dice.mean(),
        "iou": iou.mean()
    }

def find_best_threshold(probs, targets, metric="dice"):

    thresholds = np.linspace(0.05, 0.95, 19)

    best_t = 0.5
    best_score = 0.0

    for t in thresholds:
        preds = (probs > t).astype(np.float32)

        TP = np.sum(preds * targets)
        FP = np.sum(preds * (1 - targets))
        FN = np.sum((1 - preds) * targets)

        dice = (2 * TP) / (2 * TP + FP + FN + 1e-7)
        f1 = dice  

        score = dice if metric == "dice" else f1

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score

def plot_pr_curve(probs, targets, title):
    precision, recall, thresholds = precision_recall_curve(targets, probs)

    best_t, best_dice = find_best_threshold(probs, targets, metric="dice")

    idx = np.argmin(np.abs(thresholds - best_t))

    plt.figure()
    plt.plot(recall, precision, linewidth=2, label="PR curve")

    plt.scatter(
        recall[idx],
        precision[idx],
        s=80,
        marker="o",
        label=f"Best threshold = {best_t:.2f}\nDice = {best_dice:.3f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def overlay_mask(image_gray, mask, alpha=0.4):
    """
    image_gray: (H,W), uint8
    mask: (H,W), {0,255}
    """
    image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    color_mask = image.copy()
    color_mask[:, :, 2] = mask  

    return cv2.addWeighted(image, 1-alpha, color_mask, alpha, 0)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)  
model.load_state_dict(torch.load(MODEL, map_location=device))
model.to(device)

    
def evaluate_and_save(
    model,
    test_loader,
    test_root,
    do_metrics=True,
    save_pred=True,
    save_overlay=True
):
    all_probs = []
    all_targets = []
    pred_dir = os.path.join(test_root, PRED_MASK_DIR)
    overlay_dir = os.path.join(test_root, OVERLAY_DIR)

    if save_pred:
        os.makedirs(pred_dir, exist_ok=True)
    if save_overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    if do_metrics:
        metrics_sum = {k: 0.0 for k in
            ["accuracy", "precision", "recall", "f1", "dice", "iou"]
        }
        num_batches = 0

    model.eval()
    with torch.no_grad():
        for imgs, masks, names in tqdm(test_loader, desc=f"Processing {test_root}"):
            imgs = imgs.to(device)

            probs = torch.sigmoid(model(imgs))     # (B,1,H,W)
            preds = (probs > threshold).float()

            # METRICS
            if do_metrics and masks is not None:
                batch_metrics = segmentation_metrics(preds, masks)
                for k in metrics_sum:
                    metrics_sum[k] += batch_metrics[k]
                num_batches += 1
                all_probs.append(probs.cpu().numpy().ravel())
                all_targets.append(masks.cpu().numpy().ravel())
                
            # SAVE OUTPUTS 
            for i in range(preds.size(0)):
                pred_mask = preds[i].cpu().numpy().squeeze()
                pred_mask = (pred_mask * 255).astype(np.uint8)
                
                # SAVE PREDICTION MASK
                if save_pred:
                    cv2.imwrite(
                        os.path.join(pred_dir, names[i]),
                        pred_mask
                    )
                    
                # SAVE OVERLAY
                if save_overlay:
                    img_gray = imgs[i, 0].cpu().numpy()
                    img_gray = (img_gray * 0.5 + 0.5) * 255
                    img_gray = img_gray.astype(np.uint8)

                    overlay = overlay_mask(img_gray, pred_mask)
                    cv2.imwrite(
                        os.path.join(overlay_dir, names[i]),
                        overlay
                    )
    # METRICS SUMMARY
    if do_metrics:
        for k in metrics_sum:
            metrics_sum[k] /= max(num_batches, 1)

        print(f"\n RESULTS OF {test_root}")
        for k, v in metrics_sum.items():
            print(f"{k.capitalize():10s}: {v:.4f}")
            
        # Draw PR curve
        if len(all_probs) > 0 and PR_CURVE == 1 and test_root == r"D:\USTH\MLmed\anasmohammedtahir\covidqu\versions\7\Infection_Segmentation_Data\Infection_Segmentation_Data\Test\COVID-19":
            all_probs = np.concatenate(all_probs)
            all_targets = np.concatenate(all_targets)

            plot_pr_curve(
                all_probs,
                all_targets,
                title=f"PR Curve - {os.path.basename(test_root)}"
            )
        return metrics_sum

    return None


from torch.utils.data import DataLoader



all_results = {}

for test_root in test_roots:
    img_dir = os.path.join(test_root, "images")
    mask_dir = os.path.join(test_root, "infection_masks")

    test_dataset = CovidDataset(img_dir, mask_dir)

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    
    print("\n" + "="*50)
    print(f"\nEvaluating on test set: {test_root}")
    all_results[test_root] = evaluate_and_save(
        model, test_loader, test_root, 
        do_metrics=DO_METRICS,
        save_pred=SAVE_PRED_MASK,
        save_overlay=SAVE_OVERLAY
    )

    