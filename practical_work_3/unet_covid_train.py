import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# DATASET
class CovidDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path, 0)
        assert img is not None
        assert img.shape == (IMG_SIZE, IMG_SIZE)

        img = img / 255.0
        img = (img - 0.5) / 0.5
        img = np.stack([img] * 3, axis=0)
        img = torch.tensor(img, dtype=torch.float32)

        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, fname)
            mask = cv2.imread(mask_path, 0)
            mask = (mask > 0).astype(np.float32)
            mask = mask[None, :, :]
            
        else:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return img, mask, fname
    
# DATA LOADERS
train_dataset = CovidDataset(
    img_dir=r"D:\USTH\MLmed\anasmohammedtahir\covidqu\versions\7\Infection_Segmentation_Data\Infection_Segmentation_Data\Train\COVID-19\images",
    mask_dir=r"D:\USTH\MLmed\anasmohammedtahir\covidqu\versions\7\Infection_Segmentation_Data\Infection_Segmentation_Data\Train\COVID-19\infection_masks"
)

val_dataset = CovidDataset(
    img_dir=r"D:\USTH\MLmed\anasmohammedtahir\covidqu\versions\7\Infection_Segmentation_Data\Infection_Segmentation_Data\Val\COVID-19\images",
    mask_dir=r"D:\USTH\MLmed\anasmohammedtahir\covidqu\versions\7\Infection_Segmentation_Data\Infection_Segmentation_Data\Val\COVID-19\infection_masks"
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
)
print(len(train_loader), len(val_loader))

# MODEL DEFINITION
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)


dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss  = smp.losses.SoftBCEWithLogitsLoss()

def loss_fn(pred, target):
    return dice_loss(pred, target) + bce_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# TRAIN
from tqdm.notebook import tqdm
def train():
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]",
            leave=False
        )
        
        for imgs, masks, _ in train_bar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        # VALIDATION
        model.eval()
        val_loss = 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]",
            leave=False
        )
        
        with torch.no_grad():
            for imgs, masks, _ in val_bar:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        #  SAVE BEST MODEL 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "unet_covid_2.pth")
            print("✔ Best model saved")

if __name__ == "__main__":
    train()
