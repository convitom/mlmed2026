# -*- coding: utf-8 -*-
"""
Training script for False Positive Reduction Network (Stage 2)
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from config_fpr import config
from data_loader_fpr import get_dataloader
from loss_fpr import get_loss_function, MetricsCalculator


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)  # [B, 1, D, H, W]
        labels = batch['label'].to(device)  # [B]
        
        # Forward
        outputs = model(images)  # [B, 2]
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Accumulate
        epoch_loss += loss.item()
        all_preds.append(outputs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_metrics(all_preds, all_labels)
    
    avg_loss = epoch_loss / len(train_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'specificity': metrics['specificity']
    }


def validate(model, val_loader, criterion, epoch, device):
    """Validation"""
    model.eval()
    
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}')
        
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Accumulate
            epoch_loss += loss.item()
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Calculate metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_metrics(all_preds, all_labels)
    
    avg_loss = epoch_loss / len(val_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'specificity': metrics['specificity']
    }


def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False):
    """Save checkpoint to Google Drive"""
    # Create both local and Drive paths
    local_path = save_path
    drive_path = save_path.replace('/content/checkpoints_fpr', '/content/drive/MyDrive/LUNA16_checkpoints_FPR')
    
    # Create directories
    os.makedirs(local_path, exist_ok=True)
    os.makedirs(drive_path, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save to Drive (primary)
    drive_ckpt = os.path.join(drive_path, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, drive_ckpt)
    print(f"✅ Saved to Drive: {drive_ckpt}")
    
    # Also save to local (backup)
    local_ckpt = os.path.join(local_path, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save(checkpoint, local_ckpt)
    print(f"   Local backup: {local_ckpt}")
    
    # Save best model
    if is_best:
        drive_best = os.path.join(drive_path, 'best_model.pth')
        torch.save(checkpoint, drive_best)
        print(f"🎉 Best model saved to Drive: {drive_best}")
        
        local_best = os.path.join(local_path, 'best_model.pth')
        torch.save(checkpoint, local_best)
        print(f"   Local backup: {local_best}")


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = get_dataloader(
        data_dir=config.get('data_root', DATA_ROOT) if 'DATA_ROOT' in globals() else '/content/data',
        candidates_file=config.get('candidates_path', CANDIDATES_PATH) if 'CANDIDATES_PATH' in globals() else '/content/data/candidates.csv',
        subset_ids=config['train_split'],
        config=config,
        phase='train'
    )
    
    val_loader = get_dataloader(
        data_dir=config.get('data_root', DATA_ROOT) if 'DATA_ROOT' in globals() else '/content/data',
        candidates_file=config.get('candidates_path', CANDIDATES_PATH) if 'CANDIDATES_PATH' in globals() else '/content/data/candidates.csv',
        subset_ids=config['val_split'],
        config=config,
        phase='val'
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    # Import model từ file GCSAM_FPR.py
    import sys
    sys.path.append('/content')
    from GCSAM_FPR import MyModel
    
    model = MyModel(num_classes=config['num_classes'])
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Loss function
    print("\nSetting up loss function...")
    criterion = get_loss_function(config)
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion.weight = criterion.weight.to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr_stage1'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    def adjust_learning_rate(optimizer, epoch):
        if epoch < config['epoch_reduce_lr1']:
            lr = config['lr_stage1']
        elif epoch < config['epoch_reduce_lr2']:
            lr = config['lr_stage2']
        else:
            lr = config['lr_stage3']
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_f1 = 0
    best_metrics = None
    
    for epoch in range(config['start_epoch'], config['epoch']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epoch']}")
        print(f"{'='*60}")
        
        # Adjust learning rate
        current_lr = adjust_learning_rate(optimizer, epoch)
        print(f"Learning rate: {current_lr}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, device
        )
        
        print(f"\nTraining metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  F1 Score: {train_metrics['f1']:.4f}")
        print(f"  Specificity: {train_metrics['specificity']:.4f}")
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, epoch, device
        )
        
        print(f"\nValidation metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1 Score: {val_metrics['f1']:.4f}")
        print(f"  Specificity: {val_metrics['specificity']:.4f}")
        
        # Check if best model (based on F1 score)
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics
            print(f"\n🎉 New best F1 score: {best_f1:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['save_freq'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                config.get('save_path', SAVE_PATH) if 'SAVE_PATH' in globals() else '/content/checkpoints_fpr',
                is_best=is_best
            )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    if best_metrics:
        print("\nBest validation metrics:")
        print(f"  Loss: {best_metrics['loss']:.4f}")
        print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        print(f"  F1 Score: {best_metrics['f1']:.4f}")
        print(f"  Specificity: {best_metrics['specificity']:.4f}")


if __name__ == "__main__":
    main()
