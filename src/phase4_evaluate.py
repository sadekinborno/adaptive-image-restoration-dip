import os
import sys
import cv2
import numpy as np
import contextlib
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

# Silence OpenCV internal log messages
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

@contextlib.contextmanager
def suppress_c_stderr():
    """Suppresses C-level stderr output (such as libjpeg marker warnings)."""
    try:
        null_fds = os.open(os.devnull, os.O_RDWR)
        save_fds = os.dup(2)
        os.dup2(null_fds, 2)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(save_fds, 2)
            os.close(null_fds)
            os.close(save_fds)
        except Exception:
            pass

def fast_psnr(img1, img2):
    """Calculates PSNR quickly using NumPy."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10((255.0 ** 2) / mse))

def fast_ssim(img1, img2, target_size=640):
    """
    Calculates SSIM efficiently by standardizing resolution to standard benchmark scale (max 640px).
    This matches standard DIP evaluation practices and avoids 10+ hour CPU bottlenecks on 13MP images.
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    h, w = g1.shape
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        g1 = cv2.resize(g1, (new_w, new_h), interpolation=cv2.INTER_AREA)
        g2 = cv2.resize(g2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    return float(ssim(g1, g2, data_range=255))

def process_evaluation(deg_path_str, raw_dir_str, degraded_dir_str, restored_dir_str):
    deg_path = Path(deg_path_str)
    raw_dir = Path(raw_dir_str)
    degraded_dir = Path(degraded_dir_str)
    restored_dir = Path(restored_dir_str)

    base_name = deg_path.name
    rel_path = deg_path.relative_to(degraded_dir)
    
    if deg_path.stem.endswith('_blur'):
        orig_stem = deg_path.stem[:-5]
        flaw_type = "blur"
    elif deg_path.stem.endswith('_dark'):
        orig_stem = deg_path.stem[:-5]
        flaw_type = "dark"
    elif deg_path.stem.endswith('_noise'):
        orig_stem = deg_path.stem[:-6]
        flaw_type = "noise"
    else:
        orig_stem = deg_path.stem
        flaw_type = "unknown"

    orig_ext = deg_path.suffix
    orig_path = raw_dir / rel_path.parent / f"{orig_stem}{orig_ext}"
    rest_path = restored_dir / rel_path
    
    if not orig_path.exists() or not rest_path.exists():
        return None
        
    with suppress_c_stderr():
        orig_img = cv2.imread(str(orig_path))
        deg_img = cv2.imread(str(deg_path))
        rest_img = cv2.imread(str(rest_path))
    
    if orig_img is None or deg_img is None or rest_img is None:
        return None

    # Handle slight dimension mismatch if any
    if orig_img.shape != deg_img.shape:
        deg_img = cv2.resize(deg_img, (orig_img.shape[1], orig_img.shape[0]))
    if orig_img.shape != rest_img.shape:
        rest_img = cv2.resize(rest_img, (orig_img.shape[1], orig_img.shape[0]))

    deg_psnr = fast_psnr(orig_img, deg_img)
    rest_psnr = fast_psnr(orig_img, rest_img)
    deg_ssim = fast_ssim(orig_img, deg_img)
    rest_ssim = fast_ssim(orig_img, rest_img)
    
    return {
        "name": base_name,
        "flaw_type": flaw_type,
        "deg_psnr": deg_psnr,
        "rest_psnr": rest_psnr,
        "deg_ssim": deg_ssim,
        "rest_ssim": rest_ssim
    }

def main():
    raw_dir = Path("dataset/raw")
    degraded_dir = Path("data/degraded")
    restored_dir = Path("data/restored")
    csv_file = Path("data/evaluation_metrics.csv")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    raw_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        raw_paths.extend(degraded_dir.rglob(ext))
        
    seen = set()
    degraded_paths = []
    for p in raw_paths:
        canonical = str(p.resolve()).lower()
        if canonical not in seen:
            seen.add(canonical)
            degraded_paths.append(p)
        
    if not degraded_paths:
        print("No degraded images found in data/degraded.")
        return
        
    total_images = len(degraded_paths)
    print(f"Total images found for evaluation: {total_images}")
    
    # Check for already evaluated files to allow resuming
    evaluated_names = set()
    results = []
    if csv_file.exists():
        try:
            with open(csv_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    evaluated_names.add(row["name"])
                    results.append({
                        "name": row["name"],
                        "flaw_type": row["flaw_type"],
                        "deg_psnr": float(row["deg_psnr"]),
                        "rest_psnr": float(row["rest_psnr"]),
                        "deg_ssim": float(row["deg_ssim"]),
                        "rest_ssim": float(row["rest_ssim"])
                    })
            if evaluated_names:
                print(f"Found {len(evaluated_names)} already evaluated images in CSV. Resuming...")
        except Exception:
            pass

    # Filter out already evaluated
    pending_paths = [p for p in degraded_paths if p.name not in evaluated_names]
    print(f"Pending evaluation: {len(pending_paths)} images...")
    
    # Open CSV for appending new results
    csv_exists = csv_file.exists() and len(evaluated_names) > 0
    f_csv = open(csv_file, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(f_csv)
    if not csv_exists:
        csv_writer.writerow(["name", "flaw_type", "deg_psnr", "rest_psnr", "deg_ssim", "rest_ssim"])
        f_csv.flush()
        
    count = len(evaluated_names)
    
    if pending_paths:
        print(f"Starting parallel evaluation with C-warning suppression and standardized resolution...")
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    process_evaluation, 
                    str(p), 
                    str(raw_dir), 
                    str(degraded_dir), 
                    str(restored_dir)
                ) for p in pending_paths
            ]
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)
                    csv_writer.writerow([
                        res["name"], res["flaw_type"], 
                        f"{res['deg_psnr']:.2f}", f"{res['rest_psnr']:.2f}",
                        f"{res['deg_ssim']:.4f}", f"{res['rest_ssim']:.4f}"
                    ])
                    count += 1
                    
                    if count % 250 == 0 or count == total_images:
                        f_csv.flush()
                        percent = (count / total_images) * 100
                        print(f"Progress: {count}/{total_images} ({percent:.1f}%) evaluated...", flush=True)

    f_csv.close()
    
    if not results:
        print("No evaluations could be completed.")
        return
        
    # Calculate aggregate summary stats
    stats = {}
    for flaw in ["blur", "dark", "noise"]:
        flaw_res = [r for r in results if r["flaw_type"] == flaw]
        if flaw_res:
            stats[flaw] = {
                "count": len(flaw_res),
                "deg_psnr": np.mean([r["deg_psnr"] for r in flaw_res]),
                "rest_psnr": np.mean([r["rest_psnr"] for r in flaw_res]),
                "deg_ssim": np.mean([r["deg_ssim"] for r in flaw_res]),
                "rest_ssim": np.mean([r["rest_ssim"] for r in flaw_res])
            }
            
    print("\n" + "=" * 92)
    print("                      AGGREGATE PIPELINE EVALUATION RESULTS                      ")
    print("=" * 92)
    print(f"{'Flaw Type':<10} | {'Count':<7} | {'Degraded PSNR':<15} | {'Restored PSNR':<15} | {'Degraded SSIM':<15} | {'Restored SSIM':<15}")
    print("-" * 92)
    for flaw, s in stats.items():
        psnr_gain = s['rest_psnr'] - s['deg_psnr']
        ssim_gain = s['rest_ssim'] - s['deg_ssim']
        gain_str = f"({'+' if psnr_gain >= 0 else ''}{psnr_gain:.2f} dB)"
        print(f"{flaw.upper():<10} | {s['count']:<7} | {s['deg_psnr']:<15.2f} | {s['rest_psnr']:<8.2f} {gain_str:<6} | {s['deg_ssim']:<15.4f} | {s['rest_ssim']:<15.4f}")
    print("=" * 92)
    print(f"Detailed image-level metrics saved to: {csv_file.resolve()}\n")

if __name__ == "__main__":
    main()
