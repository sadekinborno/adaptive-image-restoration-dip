import os
import cv2
import numpy as np
from pathlib import Path

def calculate_local_patch_variance(gray_image, patch_size=16):
    """
    Divides the image into non-overlapping patches and calculates the variance of each.
    To isolate high-frequency sensor noise from natural image edges/textures, 
    we look at the smoothest patches (the noise floor). 
    """
    h, w = gray_image.shape
    h_patches = h // patch_size
    w_patches = w // patch_size
    
    # Crop to fit patches evenly
    trimmed = gray_image[:h_patches * patch_size, :w_patches * patch_size]
    
    # Reshape to (h_patches, w_patches, patch_size, patch_size)
    patches = trimmed.reshape(h_patches, patch_size, w_patches, patch_size).swapaxes(1, 2)
    
    # Calculate variance for each patch
    patch_vars = np.var(patches, axis=(2, 3))
    
    # A robust estimation for noise is the lower percentiles of local variances,
    # because these represent flat regions where the only variation is the noise itself.
    # We take the 10th percentile to avoid clipped/purely black patches which might have 0 variance.
    noise_variance_estimate = np.percentile(patch_vars, 10)
    
    return noise_variance_estimate

def diagnose_image_array(image):
    """Takes a BGR image array and returns diagnostic metrics."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_var = np.var(laplacian)
    global_mean = np.mean(gray)
    local_var = calculate_local_patch_variance(gray, patch_size=16)
    return {
        "laplacian_variance": float(laplacian_var),
        "global_mean_intensity": float(global_mean),
        "local_patch_variance": float(local_var)
    }

def diagnose_image(img_path):
    """Takes an image path and returns a dictionary of diagnostic metrics."""
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not read image: {img_path}")
    return diagnose_image_array(image)

def main():
    degraded_dir = Path("data/degraded")
    
    if not degraded_dir.exists():
        print(f"Directory {degraded_dir} does not exist. Run Phase 1 first.")
        return
        
    image_paths = sorted(list(degraded_dir.glob("*.jpg")))
    if not image_paths:
        print(f"No images found in {degraded_dir}")
        return
        
    print(f"Running diagnostics on {len(image_paths)} degraded images...\n")
    
    # We will just evaluate a few specific ones (blur, dark, noise) for demonstration
    sample_images = [p for p in image_paths if p.stem.startswith("100_")]
    if not sample_images:
        sample_images = image_paths[:5]
        
    for img_path in sample_images:
        metrics = diagnose_image(img_path)
        print(f"Image: {img_path.name}")
        print(f"  - Laplacian Variance (Blur)  : {metrics['laplacian_variance']:>8.2f}")
        print(f"  - Global Mean (Exposure)     : {metrics['global_mean_intensity']:>8.2f}")
        print(f"  - Local Patch Var (Noise)    : {metrics['local_patch_variance']:>8.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()
