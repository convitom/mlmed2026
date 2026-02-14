# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for LUNA16 dataset
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class LunaDataset(Dataset):
    def __init__(self, data_dir, annotations_file, subset_ids, config, phase='train'):
        """
        Args:
            data_dir: folder contains subsets (subset0, subset1,...)
            annotations_file: path to annotations.csv file
            subset_ids: List of subset IDs to use (e.g., [0,1,2,3,4,5,6,7])
            config: Configuration dictionary
            phase: 'train' or 'val'
        """
        self.data_dir = data_dir
        self.phase = phase
        self.config = config
        
        # Read annotations
        self.annotations_df = pd.read_csv(annotations_file)
        
        # Get list of CT files to use
        self.filenames = []
        for subset_id in subset_ids:
            subset_dir = os.path.join(data_dir, f'subset{subset_id}')
            if os.path.exists(subset_dir):
                # Get all .mhd files in subset
                mhd_files = [f.replace('.mhd', '') for f in os.listdir(subset_dir) if f.endswith('.mhd')]
                self.filenames.extend(mhd_files)
        
        print(f"{phase} dataset: Found {len(self.filenames)} CT scans")
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # Load CT image
        image, origin, spacing = self.load_ct(filename)
        
        # Get annotations for this scan
        scan_annotations = self.annotations_df[
            self.annotations_df['seriesuid'] == filename
        ]
        
        # Convert world coordinates to voxel coordinates
        bboxes = []
        for _, row in scan_annotations.iterrows():
            coord_world = np.array([row['coordX'], row['coordY'], row['coordZ']])
            diameter = row['diameter_mm']
            
            # Convert to voxel coordinates
            coord_voxel = self.world_to_voxel(coord_world, origin, spacing)
            
            # Bounding box: [x, y, z, diameter]
            bboxes.append([coord_voxel[0], coord_voxel[1], coord_voxel[2], diameter])
        
        bboxes = np.array(bboxes) if len(bboxes) > 0 else np.zeros((0, 4))
        
        # Crop patches from image
        if self.phase == 'train':
            sample = self.crop_patch_train(image, bboxes, filename)
        else:
            sample = self.crop_patch_val(image, bboxes, filename)
            
        return sample
    
    def load_ct(self, filename):
        """Load CT scan from .mhd file"""
        # Find file in subsets
        filepath = None
        for subset_id in range(10):
            potential_path = os.path.join(
                self.data_dir, 
                f'subset{subset_id}', 
                f'{filename}.mhd'
            )
            if os.path.exists(potential_path):
                filepath = potential_path
                break
        
        if filepath is None:
            raise FileNotFoundError(f"Cannot find {filename}.mhd in any subset")
        
        # Load using SimpleITK
        itkimage = sitk.ReadImage(filepath)
        ct_array = sitk.GetArrayFromImage(itkimage)  # Shape: [depth, height, width]
        
        origin = np.array(itkimage.GetOrigin())  # x, y, z
        spacing = np.array(itkimage.GetSpacing())  # x, y, z
        
        # Preprocessing according to paper
        ct_array = self.preprocess(ct_array)
        
        return ct_array, origin, spacing
    
    def preprocess(self, ct_array):
        """Preprocessing according to paper: 
        - Clip HU values
        - Normalize to [0, 255]
        """
        # Clip HU values
        ct_array = np.clip(ct_array, self.config['hu_min'], self.config['hu_max'])
        
        # Normalize to [0, 255]
        ct_array = (ct_array - self.config['hu_min']) / (
            self.config['hu_max'] - self.config['hu_min']
        ) * 255.0
        
        return ct_array.astype(np.float32)
    
    def world_to_voxel(self, world_coord, origin, spacing):
        """Convert world coordinates to voxel coordinates"""
        stretched_voxel_coord = np.absolute(world_coord - origin)
        voxel_coord = stretched_voxel_coord / spacing
        return voxel_coord
    
    def crop_patch_train(self, image, bboxes, filename):
        """Crop random patch for training"""
        crop_size = self.config['crop_size']
        
        d, h, w = image.shape

        # FIX: Pad if volume is smaller than crop_size
        pad_d = max(0, crop_size[0] - d)
        pad_h = max(0, crop_size[1] - h)
        pad_w = max(0, crop_size[2] - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
            d, h, w = image.shape

        
        # If there are nodules, crop around nodule with high probability
        if len(bboxes) > 0 and np.random.rand() > 0.3:
            # Select random nodule
            bbox = bboxes[np.random.randint(0, len(bboxes))]
            center = bbox[:3]
            
            # Random offset around center
            start_d = int(max(0, center[2] - crop_size[0]//2 + np.random.randint(-20, 20)))
            start_h = int(max(0, center[1] - crop_size[1]//2 + np.random.randint(-20, 20)))
            start_w = int(max(0, center[0] - crop_size[2]//2 + np.random.randint(-20, 20)))
        else:
            # Random crop
            start_d = np.random.randint(0, max(1, d - crop_size[0]))
            start_h = np.random.randint(0, max(1, h - crop_size[1]))
            start_w = np.random.randint(0, max(1, w - crop_size[2]))
        
        # Ensure not exceeding boundary
        start_d = min(start_d, d - crop_size[0])
        start_h = min(start_h, h - crop_size[1])
        start_w = min(start_w, w - crop_size[2])
        
        # Crop
        patch = image[
            start_d:start_d+crop_size[0],
            start_h:start_h+crop_size[1],
            start_w:start_w+crop_size[2]
        ]
        
        # Adjust bboxes to patch coordinates
        adjusted_bboxes = []
        for bbox in bboxes:
            x, y, z, d = bbox
            # Check if nodule is in the patch
            if (start_w <= x < start_w + crop_size[2] and
                start_h <= y < start_h + crop_size[1] and
                start_d <= z < start_d + crop_size[0]):
                
                new_bbox = [
                    x - start_w,
                    y - start_h, 
                    z - start_d,
                    d
                ]
                adjusted_bboxes.append(new_bbox)
        
        adjusted_bboxes = np.array(adjusted_bboxes) if len(adjusted_bboxes) > 0 else np.zeros((0, 4))
        
        # Data augmentation
        if self.config['augtype']['flip'] and np.random.rand() > 0.5:
            patch = np.flip(patch, axis=0).copy()
            if len(adjusted_bboxes) > 0:
                adjusted_bboxes[:, 2] = crop_size[0] - adjusted_bboxes[:, 2]
                
        if self.config['augtype']['flip'] and np.random.rand() > 0.5:
            patch = np.flip(patch, axis=1).copy()
            if len(adjusted_bboxes) > 0:
                adjusted_bboxes[:, 1] = crop_size[1] - adjusted_bboxes[:, 1]
                
        if self.config['augtype']['flip'] and np.random.rand() > 0.5:
            patch = np.flip(patch, axis=2).copy()
            if len(adjusted_bboxes) > 0:
                adjusted_bboxes[:, 0] = crop_size[2] - adjusted_bboxes[:, 0]
        
        # Add channel dimension
        patch = patch[np.newaxis, ...]  # Shape: [1, D, H, W]
        
        return {
            'image': torch.from_numpy(patch.copy()),
            'bboxes': torch.from_numpy(adjusted_bboxes.copy()),
            'filename': filename
        }
    
    def crop_patch_val(self, image, bboxes, filename):
        """Crop patches for validation - divide entire image into patches"""
        crop_size = self.config['crop_size']
        stride = crop_size[0] // 2  # 50% overlap
        
        d, h, w = image.shape

        # FIX: Pad if volume is smaller than crop_size
        pad_d = max(0, crop_size[0] - d)
        pad_h = max(0, crop_size[1] - h)
        pad_w = max(0, crop_size[2] - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='constant',
                constant_values=0
            )
            d, h, w = image.shape
        
        # Calculate number of patches needed
        patches = []
        patch_bboxes = []
        patch_coords = []  # Store coordinates for reconstruction later
        
        for start_d in range(0, d, stride):
            for start_h in range(0, h, stride):
                for start_w in range(0, w, stride):
                    end_d = min(start_d + crop_size[0], d)
                    end_h = min(start_h + crop_size[1], h)
                    end_w = min(start_w + crop_size[2], w)
                    
                    # Adjust start if needed
                    if end_d == d:
                        start_d = d - crop_size[0]
                    if end_h == h:
                        start_h = h - crop_size[1]
                    if end_w == w:
                        start_w = w - crop_size[2]
                    
                    # Crop patch
                    patch = image[
                        start_d:start_d+crop_size[0],
                        start_h:start_h+crop_size[1],
                        start_w:start_w+crop_size[2]
                    ]
                    
                    # Adjust bboxes
                    adjusted_bboxes = []
                    for bbox in bboxes:
                        x, y, z, diameter = bbox
                        if (start_w <= x < start_w + crop_size[2] and
                            start_h <= y < start_h + crop_size[1] and
                            start_d <= z < start_d + crop_size[0]):
                            
                            new_bbox = [
                                x - start_w,
                                y - start_h,
                                z - start_d,
                                diameter
                            ]
                            adjusted_bboxes.append(new_bbox)
                    
                    adjusted_bboxes = np.array(adjusted_bboxes) if len(adjusted_bboxes) > 0 else np.zeros((0, 4))
                    
                    patch = patch[np.newaxis, ...]  # Add channel
                    patches.append(torch.from_numpy(patch.copy()))
                    patch_bboxes.append(torch.from_numpy(adjusted_bboxes.copy()))
                    patch_coords.append([start_d, start_h, start_w])
        
        return {
            'images': patches,  # List of patches
            'bboxes': patch_bboxes,  # List of bbox arrays
            'coords': patch_coords,  # Patch coordinates
            'filename': filename,
            'full_shape': image.shape
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    if 'images' in batch[0]:  # Validation batch
        return batch  # Return as is for validation
    else:  # Training batch
        images = torch.stack([item['image'] for item in batch])
        bboxes = [item['bboxes'] for item in batch]
        filenames = [item['filename'] for item in batch]
        
        return {
            'images': images,
            'bboxes': bboxes,
            'filenames': filenames
        }


if __name__ == "__main__":
    # Test dataset
    from config_cnd import config
    
    dataset = LunaDataset(
        data_dir='/content/data',
        annotations_file='/content/data/annotations.csv',
        subset_ids=[0],
        config=config,
        phase='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Number of bboxes: {len(sample['bboxes'])}")
