# -*- coding: utf-8 -*-
"""
Configuration file for lung nodule detection training
"""

# Paths
DATA_ROOT = '/content/data'  # Root folder containing data on Colab
RAW_DATA_PATH = '/content/data/subset0'  # Directory containing .raw and .mhd files
ANNOTATION_PATH = '/content/data/annotations.csv'  # Annotations file
SAVE_PATH = '/content/drive/MyDrive/checkpoints'  # Directory to save models

# Model parameters
config = {}
config['anchors'] = [5.0, 10.0, 20.0]  # Anchor sizes for detection
config['channel_size'] = 1  # Number of channels (CT is 1 channel)
config['crop_size'] = [128, 128, 128]  # Crop size for training
config['stride'] = 4  # Stride when dividing image into patches
config['max_stride'] = 16  # Maximum stride
config['num_neg'] = 800  # Number of negative samples per epoch
config['th_neg'] = 0.02  # Threshold to determine negative sample
config['th_pos_train'] = 0.3  # Threshold to determine positive sample during training
config['th_pos_val'] = 0.1  # Threshold during validation
config['num_hard'] = 2  # Number of hard negative samples
config['bound_size'] = 12  # Boundary size
config['reso'] = 1  # Resolution
config['sizelim'] = 3.0  # Size limit for nodules (mm)
config['sizelim2'] = 30  # Max size limit
config['sizelim3'] = 40  # 
config['aug_scale'] = True  # Data augmentation: scale
config['r_rand_crop'] = 0.3  # Random crop ratio
config['pad_value'] = 170  # Padding value
config['augtype'] = {'flip': True, 'rotate': True, 'swap': False}  # Augmentation types

# Training parameters
config['lr_stage1'] = 0.01  # Learning rate for stage 1
config['lr_stage2'] = 0.001  # Learning rate for first reduction
config['lr_stage3'] = 0.0001  # Learning rate for second reduction
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['epoch'] = 200  # Total number of epochs
config['epoch_rcnn'] = 50  # Epoch to reduce lr first time
config['epoch_rcnn2'] = 100  # Epoch to reduce lr second time
config['batch_size'] = 4
config['workers'] = 4  # Number of workers for DataLoader
config['start_epoch'] = 0
config['save_freq'] = 5  # Save checkpoint every N epochs

# Dataset split
config['train_split'] = [0, 1, 2, 3, 4, 5, 6, 7]  # Subset IDs for training (8/10)
config['val_split'] = [8, 9]  # Subset IDs for validation (2/10)

# Preprocessing parameters (according to paper)
config['hu_min'] = -1200
config['hu_max'] = 600
config['normalize_min'] = 0
config['normalize_max'] = 255
config['fill_value'] = 170  # Fill value for area outside lung

# Device
config['n_gpu'] = 1  # Number of GPUs to use
config['device'] = 'cuda'

print("Configuration loaded successfully!")
