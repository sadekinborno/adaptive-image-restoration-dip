import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
from pathlib import Path

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Import the diagnostic function from phase 2
from phase2_diagnostics import diagnose_image_array

# Define thresholds based on Phase 2 observations
BLUR_THRESHOLD = 300.0
DARK_THRESHOLD = 80.0
NOISE_THRESHOLD = 150.0

def restore_blur(image):
    """Applies Unsharp Masking to extract and boost high-frequency details."""
    gaussian = cv2.GaussianBlur(image, (9, 9), 10.0)
    restored = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    return restored

def restore_dark(image):
    """Applies CLAHE (Contrast-Limited Adaptive Histogram Equalization) using an 8x8 grid."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_restored = cv2.merge((l_clahe, a, b))
    restored = cv2.cvtColor(lab_restored, cv2.COLOR_LAB2BGR)
    return restored

def restore_noise(image):
    """Applies a 5x5 Median Filter to remove high-frequency noise."""
    return cv2.medianBlur(image, 5)

def process_image(img_path, degraded_dir, restored_dir):
    rel_path = img_path.relative_to(degraded_dir)
    output_path = restored_dir / rel_path
    
    # If already restored, skip reading
    if output_path.exists():
        return "EXISTING"

    image = cv2.imread(str(img_path))
    if image is None:
        return None
        
    try:
        metrics = diagnose_image_array(image)
    except Exception:
        return None
        
    if metrics["laplacian_variance"] < BLUR_THRESHOLD:
        flaw = "BLUR"
        restored_image = restore_blur(image)
    elif metrics["local_patch_variance"] > NOISE_THRESHOLD:
        flaw = "NOISE"
        restored_image = restore_noise(image)
    elif metrics["global_mean_intensity"] < DARK_THRESHOLD:
        flaw = "DARK"
        restored_image = restore_dark(image)
    else:
        flaw = "NONE/UNKNOWN"
        restored_image = image  
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), restored_image)
    
    return flaw

def main():
    degraded_dir = Path("data/degraded")
    restored_dir = Path("data/restored")
    
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        image_paths.extend(degraded_dir.rglob(ext))
        
    if not image_paths:
        print(f"No degraded images found in {degraded_dir}")
        return
        
    print(f"Routing and Restoring {len(image_paths)} images using multiprocessing...\n")
    
    flaw_counts = {"BLUR": 0, "DARK": 0, "NOISE": 0, "NONE/UNKNOWN": 0, "EXISTING": 0}
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, p, degraded_dir, restored_dir) for p in image_paths]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                flaw_counts[result] = flaw_counts.get(result, 0) + 1
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")

    print("\nPhase 3 complete! All restored images saved.")
    print(f"Restoration breakdown: {flaw_counts}")

if __name__ == "__main__":
    main()
