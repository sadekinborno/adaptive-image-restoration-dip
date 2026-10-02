#!/usr/bin/env python3
"""
Adaptive Image Quality Assessment and Restoration System
Course: Digital Image Processing (CSE 4883)
Pre-processor pipeline for smart waste bin camera systems.
Strictly classical Digital Image Processing (no deep learning/ML models).
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
from pathlib import Path

# Threshold defaults established during benchmark tuning
DEFAULT_BLUR_THRESH = 300.0
DEFAULT_DARK_THRESH = 80.0
DEFAULT_NOISE_THRESH = 150.0

def calculate_local_patch_variance(gray_image, patch_size=16):
    """
    Divides image into non-overlapping patches and estimates the noise floor
    from the 10th percentile of local patch variances.
    """
    h, w = gray_image.shape
    h_patches, w_patches = h // patch_size, w // patch_size
    if h_patches == 0 or w_patches == 0:
        return float(np.var(gray_image))
        
    trimmed = gray_image[:h_patches * patch_size, :w_patches * patch_size]
    patches = trimmed.reshape(h_patches, patch_size, w_patches, patch_size).swapaxes(1, 2)
    patch_vars = np.var(patches, axis=(2, 3))
    return float(np.percentile(patch_vars, 10))

def diagnose_image(image_bgr):
    """
    Computes mathematical flaw metrics:
    - Laplacian variance (3x3 kernel) -> Blur detection
    - Global mean intensity -> Underexposure detection
    - Local patch variance -> High-frequency sensor noise detection
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_var = float(np.var(laplacian))
    mean_intensity = float(np.mean(gray))
    patch_var = calculate_local_patch_variance(gray, patch_size=16)
    
    return {
        "laplacian_variance": laplacian_var,
        "global_mean_intensity": mean_intensity,
        "local_patch_variance": patch_var
    }

def restore_blur(image_bgr):
    """Applies Unsharp Masking to boost high-frequency details."""
    gaussian = cv2.GaussianBlur(image_bgr, (9, 9), 10.0)
    return cv2.addWeighted(image_bgr, 1.5, gaussian, -0.5, 0)

def restore_dark(image_bgr):
    """Applies CLAHE on the Lightness channel using an 8x8 grid."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_restored = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_restored, a, b)), cv2.COLOR_LAB2BGR)

def restore_noise(image_bgr):
    """Applies a 5x5 Median Filter to remove high-frequency noise."""
    return cv2.medianBlur(image_bgr, 5)

def route_and_restore(image_bgr, metrics, blur_th, dark_th, noise_th):
    """Evaluates metrics and routes to the appropriate restoration filter."""
    if metrics["laplacian_variance"] < blur_th:
        flaw = "BLUR"
        restored = restore_blur(image_bgr)
    elif metrics["local_patch_variance"] > noise_th:
        flaw = "NOISE"
        restored = restore_noise(image_bgr)
    elif metrics["global_mean_intensity"] < dark_th:
        flaw = "DARK"
        restored = restore_dark(image_bgr)
    else:
        flaw = "NONE"
        restored = image_bgr.copy()
        
    return flaw, restored

def extract_canny(image_bgr):
    """Adaptive Canny edge detection."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    return cv2.Canny(gray, lower, upper)

def process_file(file_path, output_dir, blur_th, dark_th, noise_th, save_edges=False):
    img = cv2.imread(str(file_path))
    if img is None:
        print(f"[ERROR] Could not decode image: {file_path}")
        return None

    metrics = diagnose_image(img)
    flaw, restored = route_and_restore(img, metrics, blur_th, dark_th, noise_th)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / file_path.name
    cv2.imwrite(str(out_file), restored)
    
    if save_edges:
        edge_dir = out_dir / "edges"
        edge_dir.mkdir(parents=True, exist_ok=True)
        edges = extract_canny(restored)
        cv2.imwrite(str(edge_dir / f"edge_{file_path.name}"), edges)
        
    return {
        "file": file_path.name,
        "diagnosed_flaw": flaw,
        "metrics": metrics,
        "output_path": str(out_file)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Image Quality Assessment and Restoration Pipeline (CSE 4883 DIP Project)"
    )
    parser.add_argument("--input", "-i", default="/app/input", help="Path to input image or directory (default: /app/input)")
    parser.add_argument("--output", "-o", default="/app/output", help="Path to output directory (default: /app/output)")
    parser.add_argument("--blur-thresh", type=float, default=DEFAULT_BLUR_THRESH, help="Laplacian variance blur threshold")
    parser.add_argument("--dark-thresh", type=float, default=DEFAULT_DARK_THRESH, help="Global mean underexposure threshold")
    parser.add_argument("--noise-thresh", type=float, default=DEFAULT_NOISE_THRESH, help="Local patch variance noise threshold")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Process at most N images (ideal for quick tests/demos)")
    parser.add_argument("--save-edges", action="store_true", help="Also extract and save Canny edge maps")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    # Collect images
    if input_path.is_file():
        image_files = [input_path]
    else:
        raw_list = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            raw_list.extend(input_path.rglob(ext))
        # Deduplicate paths
        seen = set()
        image_files = []
        for p in raw_list:
            canonical = str(p.resolve()).lower()
            if canonical not in seen:
                seen.add(canonical)
                image_files.append(p)

    if not image_files:
        print(f"[WARNING] No valid image files found in {input_path}")
        sys.exit(0)

    if args.limit and len(image_files) > args.limit:
        print(f"[*] Limiting processing to the first {args.limit} image(s) (--limit {args.limit})")
        image_files = image_files[:args.limit]

    print(f"================================================================================")
    print(f" Adaptive Image Restoration Pipeline (Docker Container Service)")
    print(f" Input: {input_path} ({len(image_files)} image(s)) | Output: {output_path}")
    print(f" Thresholds: Blur < {args.blur_thresh} | Dark < {args.dark_thresh} | Noise > {args.noise_thresh}")
    print(f"================================================================================")

    results = []
    counts = {"BLUR": 0, "DARK": 0, "NOISE": 0, "NONE": 0}
    
    for idx, f in enumerate(image_files, 1):
        res = process_file(f, output_path, args.blur_thresh, args.dark_thresh, args.noise_thresh, args.save_edges)
        if res:
            results.append(res)
            counts[res["diagnosed_flaw"]] += 1
            if not args.json:
                print(f"[{idx}/{len(image_files)}] {res['file']:<25} -> Diagnosed: {res['diagnosed_flaw']:<6} | "
                      f"Laplacian: {res['metrics']['laplacian_variance']:<7.1f} | "
                      f"Mean: {res['metrics']['global_mean_intensity']:<5.1f} | "
                      f"Noise: {res['metrics']['local_patch_variance']:<6.1f}")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"================================================================================")
        print(f" Processing Complete! Summary: {counts}")
        print(f" Restored images saved to: {output_path.resolve()}")
        print(f"================================================================================")

if __name__ == "__main__":
    main()
