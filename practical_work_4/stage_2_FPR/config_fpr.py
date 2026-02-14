# -*- coding: utf-8 -*-
"""
Configuration file for False Positive Reduction Network (Stage 2)
"""

# Paths
DATA_ROOT = '/content/data'
CANDIDATES_PATH = '/content/data/candidates.csv'  # File candidates
SAVE_PATH = '/content/checkpoints_fpr'

# Model parameters
config = {}
config['crop_size'] = [32, 32, 32]  # Kích thước crop xung quanh candidate
config['num_classes'] = 2  # Binary classification: nodule vs non-nodule

# Preprocessing parameters
config['hu_min'] = -1200
config['hu_max'] = 600
config['normalize_min'] = 0
config['normalize_max'] = 255

# Training parameters
config['lr_stage1'] = 0.01
config['lr_stage2'] = 0.001
config['lr_stage3'] = 0.0001
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['epoch'] = 50
config['epoch_reduce_lr1'] = 15  # Giảm lr lần 1
config['epoch_reduce_lr2'] = 30  # Giảm lr lần 2
config['batch_size'] = 8
config['workers'] = 0
config['start_epoch'] = 0
config['save_freq'] = 5

# Dataset split (10-fold cross validation)
config['train_split'] = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 subsets for training
config['val_split'] = [8, 9]  # 2 subsets for validation

# Class imbalance handling
# Theo paper: positive samples được augment để tăng lên 31,140
# Negative samples chỉ lấy ~3% (22,650 samples)
config['pos_augmentation_factor'] = 20  # Số lần augment positive samples
config['neg_sample_ratio'] = 0.03  # Tỷ lệ negative samples giữ lại

# Loss function - sử dụng weighted loss do class imbalance
# Class weights: [weight_for_class_0, weight_for_class_1]
# Tính dựa trên tỷ lệ inverse của số samples
# Nếu có 100 negatives và 10 positives -> weights = [10/110, 100/110] = [0.09, 0.91]
config['use_weighted_loss'] = True
config['pos_weight'] = 10.0  # Weight cho positive class (điều chỉnh dựa trên tỷ lệ thực tế)

# Focal Loss parameters (alternative to weighted BCE)
config['use_focal_loss'] = False  # Set True nếu muốn dùng Focal Loss
config['focal_alpha'] = 0.25
config['focal_gamma'] = 2.0

# Data augmentation (chỉ cho positive samples)
config['augmentation'] = {
    'flip': True,
    'rotate': True,
    'scale': False,
    'noise': False
}

# Device
config['device'] = 'cuda'

print("FPR Configuration loaded successfully!")
