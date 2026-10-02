import os
import cv2
import numpy as np
from pathlib import Path

def apply_defocus_blur(image, kernel_size=(15, 15), sigma=5):
    """Applies a Gaussian Blur to simulate defocus."""
    return cv2.GaussianBlur(image, kernel_size, sigma)

def apply_underexposure(image, gamma=2.5):
    """Applies a power-law (Gamma) transformation for underexposure (gamma > 1)."""
    # Normalize to [0, 1]
    img_float = image.astype(np.float32) / 255.0
    # Apply gamma > 1 for darkening
    dark_img = np.power(img_float, gamma)
    dark_img = np.clip(dark_img * 255.0, 0, 255).astype(np.uint8)
    return dark_img

def apply_sensor_noise(image, mean=0, sigma=25):
    """Applies additive Gaussian noise to simulate sensor noise."""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_img = image.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

def process_image(img_path, input_dir, output_dir):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
        
    rel_path = img_path.relative_to(input_dir)
    base_name = img_path.stem
    ext = img_path.suffix
    
    # Target directory preserving structure
    target_dir = output_dir / rel_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply degradations
    img_blur = apply_defocus_blur(img, kernel_size=(21, 21), sigma=7)
    img_dark = apply_underexposure(img, gamma=3.0)
    img_noise = apply_sensor_noise(img, sigma=40)
    
    # Save output images
    cv2.imwrite(str(target_dir / f"{base_name}_blur{ext}"), img_blur)
    cv2.imwrite(str(target_dir / f"{base_name}_dark{ext}"), img_dark)
    cv2.imwrite(str(target_dir / f"{base_name}_noise{ext}"), img_noise)
    return True

from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    input_dir = Path("dataset/raw")
    output_dir = Path("data/degraded")
    
    # Find all images in input directory
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        image_paths.extend(input_dir.rglob(ext))
        
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Starting degradation process using multiprocessing...")
    
    success_count = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, p, input_dir, output_dir) for p in image_paths]
        
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success_count += 1
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")
                
    print(f"Degradation complete. {success_count} original images processed (generating {success_count*3} degraded images).")

if __name__ == "__main__":
    main()
