# -*- coding: utf-8 -*-
"""
Data loader for False Positive Reduction Network (Stage 2)
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import rotate as scipy_rotate
import warnings
warnings.filterwarnings('ignore')


class CandidateDataset(Dataset):
    def __init__(self, data_dir, candidates_file, subset_ids, config, phase='train'):
        """
        Args:
            data_dir: Thư mục chứa các subset CT scans
            candidates_file: File CSV chứa candidates (seriesuid, coordX, coordY, coordZ, class)
            subset_ids: List các subset ID cần dùng
            config: Dictionary cấu hình
            phase: 'train' hoặc 'val'
        """
        self.data_dir = data_dir
        self.config = config
        self.phase = phase
        self.crop_size = config['crop_size']
        
        # Đọc candidates file
        self.candidates_df = pd.read_csv(candidates_file)
        print(f"Total candidates in file: {len(self.candidates_df)}")
        
        # Lấy danh sách CT scans trong các subset đang dùng
        available_scans = set()
        for subset_id in subset_ids:
            subset_dir = os.path.join(data_dir, f'subset{subset_id}')
            if os.path.exists(subset_dir):
                mhd_files = [f.replace('.mhd', '') for f in os.listdir(subset_dir) 
                           if f.endswith('.mhd')]
                available_scans.update(mhd_files)
        
        # Filter candidates chỉ trong các scans có sẵn
        self.candidates_df = self.candidates_df[
            self.candidates_df['seriesuid'].isin(available_scans)
        ].reset_index(drop=True)
        
        print(f"Candidates after filtering by available scans: {len(self.candidates_df)}")
        
        # Separate positive and negative samples
        self.pos_samples = self.candidates_df[self.candidates_df['class'] == 1].reset_index(drop=True)
        self.neg_samples = self.candidates_df[self.candidates_df['class'] == 0].reset_index(drop=True)
        
        print(f"Positive samples: {len(self.pos_samples)}")
        print(f"Negative samples: {len(self.neg_samples)}")
        
        # Handle class imbalance
        if phase == 'train':
            # Augment positive samples
            pos_aug_factor = config.get('pos_augmentation_factor', 20)
            self.pos_samples = pd.concat([self.pos_samples] * pos_aug_factor, 
                                        ignore_index=True)
            
            # Downsample negative samples
            neg_ratio = config.get('neg_sample_ratio', 0.03)
            n_neg_samples = int(len(self.neg_samples) * neg_ratio)
            self.neg_samples = self.neg_samples.sample(n=n_neg_samples, 
                                                       random_state=42).reset_index(drop=True)
            
            print(f"\nAfter balancing (training):")
            print(f"  Positive samples: {len(self.pos_samples)}")
            print(f"  Negative samples: {len(self.neg_samples)}")
        
        # Combine positive and negative samples
        self.samples = pd.concat([self.pos_samples, self.neg_samples], 
                                ignore_index=True)
        
        # Shuffle
        if phase == 'train':
            self.samples = self.samples.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"{phase} dataset: {len(self.samples)} samples")
        
        # Cache cho CT scans để tránh load lại nhiều lần
        self.ct_cache = {}
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        
        seriesuid = row['seriesuid']
        coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
        label = int(row['class'])
        
        # Load CT scan
        ct_array, origin, spacing = self.load_ct(seriesuid)
        
        # Convert world coordinates to voxel coordinates
        voxel_coord = self.world_to_voxel(coord, origin, spacing)
        
        # Crop patch centered at candidate
        patch = self.crop_patch(ct_array, voxel_coord)
        
        # Data augmentation (chỉ cho positive samples trong training)
        if self.phase == 'train' and label == 1:
            patch = self.augment(patch)
        
        # Convert to tensor
        patch = torch.from_numpy(patch[np.newaxis, ...].copy()).float()  # Add channel dim
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': patch,
            'label': label,
            'seriesuid': seriesuid,
            'coord': coord
        }
    
    def load_ct(self, seriesuid):
        """Load CT scan, với caching"""
        if seriesuid in self.ct_cache:
            return self.ct_cache[seriesuid]
        
        # Tìm file
        filepath = None
        for subset_id in range(10):
            potential_path = os.path.join(
                self.data_dir, 
                f'subset{subset_id}', 
                f'{seriesuid}.mhd'
            )
            if os.path.exists(potential_path):
                filepath = potential_path
                break
        
        if filepath is None:
            raise FileNotFoundError(f"Cannot find {seriesuid}.mhd")
        
        # Load
        itkimage = sitk.ReadImage(filepath)
        ct_array = sitk.GetArrayFromImage(itkimage)  # [D, H, W]
        origin = np.array(itkimage.GetOrigin())
        spacing = np.array(itkimage.GetSpacing())
        
        # Preprocessing
        ct_array = self.preprocess(ct_array)
        
        # Cache (giới hạn cache size nếu cần)
        if len(self.ct_cache) < 10:  # Cache tối đa 10 CT scans
            self.ct_cache[seriesuid] = (ct_array, origin, spacing)
        
        return ct_array, origin, spacing
    
    def preprocess(self, ct_array):
        """Preprocessing giống stage 1"""
        ct_array = np.clip(ct_array, self.config['hu_min'], self.config['hu_max'])
        ct_array = (ct_array - self.config['hu_min']) / (
            self.config['hu_max'] - self.config['hu_min']
        ) * 255.0
        return ct_array.astype(np.float32)
    
    def world_to_voxel(self, world_coord, origin, spacing):
        """Convert world coordinates to voxel coordinates"""
        voxel_coord = np.absolute(world_coord - origin) / spacing
        return voxel_coord
    
    def crop_patch(self, ct_array, voxel_coord):
        """Crop 32x32x32 patch centered at voxel_coord"""
        D, H, W = ct_array.shape
        crop_d, crop_h, crop_w = self.crop_size
        
        # Calculate crop boundaries
        x, y, z = voxel_coord
        x, y, z = int(x), int(y), int(z)
        
        start_d = max(0, z - crop_d // 2)
        end_d = start_d + crop_d
        
        start_h = max(0, y - crop_h // 2)
        end_h = start_h + crop_h
        
        start_w = max(0, x - crop_w // 2)
        end_w = start_w + crop_w
        
        # Adjust if exceeds boundaries
        if end_d > D:
            end_d = D
            start_d = max(0, D - crop_d)
        
        if end_h > H:
            end_h = H
            start_h = max(0, H - crop_h)
        
        if end_w > W:
            end_w = W
            start_w = max(0, W - crop_w)
        
        # Crop
        patch = ct_array[start_d:end_d, start_h:end_h, start_w:end_w]
        
        # Pad if needed
        if patch.shape != tuple(self.crop_size):
            padded = np.zeros(self.crop_size, dtype=np.float32)
            d, h, w = patch.shape
            padded[:d, :h, :w] = patch
            patch = padded
        
        return patch
    
    def augment(self, patch):
        """Data augmentation for training"""
        aug_config = self.config['augmentation']
        
        # Random flip
        if aug_config['flip'] and np.random.rand() > 0.5:
            patch = np.flip(patch, axis=0).copy()
        
        if aug_config['flip'] and np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1).copy()
        
        if aug_config['flip'] and np.random.rand() > 0.5:
            patch = np.flip(patch, axis=2).copy()
        
        # Random rotation
        if aug_config['rotate'] and np.random.rand() > 0.5:
            angle = np.random.uniform(-10, 10)
            axes = [(0, 1), (0, 2), (1, 2)][np.random.randint(0, 3)]
            patch = scipy_rotate(patch, angle, axes=axes, reshape=False, order=1)
        
        return patch


def get_dataloader(data_dir, candidates_file, subset_ids, config, phase='train'):
    """Create dataloader"""
    dataset = CandidateDataset(
        data_dir=data_dir,
        candidates_file=candidates_file,
        subset_ids=subset_ids,
        config=config,
        phase=phase
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=(phase == 'train'),
        num_workers=config['workers'],
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset
    from config_fpr import config
    
    dataset = CandidateDataset(
        data_dir='/content/data',
        candidates_file='/content/data/candidates.csv',
        subset_ids=[0],
        config=config,
        phase='train'
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  SeriesUID: {sample['seriesuid']}")
