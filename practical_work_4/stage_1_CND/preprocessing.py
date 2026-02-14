# -*- coding: utf-8 -*-
"""
Preprocessing for LUNA16 dataset following the paper methodology
Reference: Figure 6 in the paper
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import morphology, measure
import warnings
warnings.filterwarnings('ignore')


class LungPreprocessor:
    """
    Preprocessing pipeline theo paper:
    1. Extract lung mask
    2. HU clipping [-1200, 600]
    3. Normalize to [0, 255]
    4. Apply mask và fill background với 170
    """
    
    def __init__(self, hu_min=-1200, hu_max=600, fill_value=170):
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.fill_value = fill_value
        
    def preprocess(self, ct_array):
        """
        Full preprocessing pipeline
        
        Args:
            ct_array: Raw CT array [D, H, W] với HU values
            
        Returns:
            Preprocessed CT array [D, H, W] normalized to [0, 255]
        """
        # Step 1: Extract lung mask
        lung_mask = self.extract_lung_mask(ct_array)
        
        # Step 2: HU clipping
        clipped = np.clip(ct_array, self.hu_min, self.hu_max)
        
        # Step 3: Normalize to [0, 255]
        normalized = (clipped - self.hu_min) / (self.hu_max - self.hu_min) * 255.0
        normalized = normalized.astype(np.float32)
        
        # Step 4: Apply mask và fill background
        # Multiply with mask to keep only lung region
        processed = normalized * lung_mask
        
        # Fill non-lung region với fill_value (170 theo paper)
        processed[lung_mask == 0] = self.fill_value
        
        return processed
    
    def extract_lung_mask(self, ct_array):
        """
        Extract lung mask theo Figure 6 trong paper
        
        Steps:
        a) Raw CT image
        b) Threshold at -600 HU
        c) Remove non-lung regions (connected component analysis)
        d) Erosion + hole filling
        e) Convex hull processing + dilation (10 iterations)
        f) Final processed image
        
        Args:
            ct_array: [D, H, W]
            
        Returns:
            lung_mask: [D, H, W] binary mask (0 or 1)
        """
        # Initialize mask
        lung_mask = np.zeros_like(ct_array, dtype=np.uint8)
        
        # Process slice by slice (vì lung mask chủ yếu làm trên axial slices)
        for i in range(ct_array.shape[0]):
            slice_img = ct_array[i]
            
            # (b) Threshold at -600 HU
            binary = slice_img >= -600
            binary = binary.astype(np.uint8)
            
            # (c) Remove non-lung regions connected to background
            # Label connected components
            labels = measure.label(binary, connectivity=2)
            
            # Identify background label (connected to edges)
            background_label = labels[0, 0]
            
            # Remove background
            binary[labels == background_label] = 0
            
            # (d) Erosion to remove small noise
            binary = ndimage.binary_erosion(binary, iterations=2)
            
            # Hole filling - fill small holes inside lung
            binary = ndimage.binary_fill_holes(binary)
            
            # (e) Convex hull processing để fill gaps
            # Chỉ apply nếu có lung region
            if np.sum(binary) > 0:
                # Get all non-zero regions
                regions = measure.regionprops(measure.label(binary))
                
                # Apply convex hull to each region
                for region in regions:
                    # Get convex hull
                    convex_hull = morphology.convex_hull_image(
                        binary[region.bbox[0]:region.bbox[2], 
                               region.bbox[1]:region.bbox[3]]
                    )
                    binary[region.bbox[0]:region.bbox[2], 
                           region.bbox[1]:region.bbox[3]] = convex_hull
            
            # Binary dilation (10 iterations theo paper)
            binary = ndimage.binary_dilation(binary, iterations=10)
            
            lung_mask[i] = binary
        
        return lung_mask.astype(np.float32)
    
    def simple_preprocess(self, ct_array):
        """
        Simplified preprocessing (không dùng lung mask)
        Chỉ clip HU và normalize
        
        Dùng khi muốn train nhanh hoặc test
        """
        # HU clipping
        clipped = np.clip(ct_array, self.hu_min, self.hu_max)
        
        # Normalize to [0, 255]
        normalized = (clipped - self.hu_min) / (self.hu_max - self.hu_min) * 255.0
        
        return normalized.astype(np.float32)


def load_and_preprocess_ct(filepath, use_lung_mask=True):
    """
    Load CT scan và preprocess
    
    Args:
        filepath: Path to .mhd file
        use_lung_mask: Có dùng lung mask extraction không
        
    Returns:
        processed_ct: Preprocessed CT array
        origin: Origin coordinates
        spacing: Voxel spacing
    """
    # Load CT
    itkimage = sitk.ReadImage(filepath)
    ct_array = sitk.GetArrayFromImage(itkimage)  # [D, H, W]
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    
    # Preprocess
    preprocessor = LungPreprocessor()
    
    if use_lung_mask:
        print("Using full preprocessing with lung mask extraction...")
        processed = preprocessor.preprocess(ct_array)
    else:
        print("Using simple preprocessing (no lung mask)...")
        processed = preprocessor.simple_preprocess(ct_array)
    
    return processed, origin, spacing


def visualize_preprocessing_steps(ct_array, save_path=None):
    """
    Visualize các bước preprocessing như Figure 6 trong paper
    """
    import matplotlib.pyplot as plt
    
    preprocessor = LungPreprocessor()
    
    # Get middle slice
    mid_slice = ct_array.shape[0] // 2
    original_slice = ct_array[mid_slice]
    
    # Step-by-step preprocessing
    # (a) Raw image
    raw = original_slice
    
    # (b) Threshold at -600
    threshold = (original_slice >= -600).astype(np.uint8)
    
    # (c) Remove background
    labels = measure.label(threshold, connectivity=2)
    background_label = labels[0, 0]
    no_background = threshold.copy()
    no_background[labels == background_label] = 0
    
    # (d) Erosion + hole filling
    eroded = ndimage.binary_erosion(no_background, iterations=2)
    filled = ndimage.binary_fill_holes(eroded)
    
    # (e) Convex hull + dilation
    regions = measure.regionprops(measure.label(filled))
    convex = filled.copy()
    for region in regions:
        hull = morphology.convex_hull_image(
            filled[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]]
        )
        convex[region.bbox[0]:region.bbox[2], region.bbox[1]:region.bbox[3]] = hull
    
    dilated = ndimage.binary_dilation(convex, iterations=10)
    
    # (f) Final processed
    clipped = np.clip(original_slice, -1200, 600)
    normalized = (clipped - (-1200)) / (600 - (-1200)) * 255.0
    final = normalized * dilated
    final[dilated == 0] = 170
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(raw, cmap='gray')
    axes[0, 0].set_title('(a) Raw CT Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(threshold, cmap='gray')
    axes[0, 1].set_title('(b) Threshold at -600 HU')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(no_background, cmap='gray')
    axes[0, 2].set_title('(c) Remove Background')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(filled, cmap='gray')
    axes[1, 0].set_title('(d) Erosion + Hole Filling')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(dilated, cmap='gray')
    axes[1, 1].set_title('(e) Convex Hull + Dilation')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(final, cmap='gray')
    axes[1, 2].set_title('(f) Final Processed')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Example usage
    import os
    
    # Path to a CT scan
    ct_path = '/content/data/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.xxx.mhd'
    
    if os.path.exists(ct_path):
        print("Loading CT scan...")
        processed, origin, spacing = load_and_preprocess_ct(ct_path, use_lung_mask=True)
        
        print(f"Processed CT shape: {processed.shape}")
        print(f"Value range: [{processed.min():.2f}, {processed.max():.2f}]")
        print(f"Origin: {origin}")
        print(f"Spacing: {spacing}")
        
        # Visualize
        itkimage = sitk.ReadImage(ct_path)
        raw_ct = sitk.GetArrayFromImage(itkimage)
        visualize_preprocessing_steps(raw_ct, save_path='/content/preprocessing_steps.png')
    else:
        print(f"File not found: {ct_path}")
        print("Please update the path to an actual CT scan file")
