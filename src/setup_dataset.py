import os
import zipfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_zip(zip_path, extract_to):
    """Extracts a zip file to the specified directory."""
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted: {zip_path.name}")
    except Exception as e:
        print(f"Failed to extract {zip_path.name}: {e}")

def copy_file(src, dst_dir):
    """Copies a file to the specified directory."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src.name
    if not dst_path.exists():
        shutil.copy2(src, dst_path)
    return dst_path

def main():
    bdwaste_raw = Path("datasets/raw/BDWaste A comprehensive image dataset of digestible and indigestible waste in Bangladesh/BDWaste")
    ugv_raw = Path("datasets/raw/UGV-NBWASTE An Oriented Non-Biodegradable Waste Dataset in Bangladesh")
    
    target_dir = Path("dataset/raw")
    
    print("Setting up unified dataset directory...")
    
    # 1. Extract BDWaste zip files
    zip_files = list(bdwaste_raw.rglob("*.zip"))
    print(f"Found {len(zip_files)} zip files in BDWaste.")
    
    # Extract sequentially to avoid massive disk I/O bottlenecks, or use limited parallel workers
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for zip_file in zip_files:
            # Recreate relative directory structure
            rel_path = zip_file.relative_to(bdwaste_raw).parent
            extract_dir = target_dir / "BDWaste" / rel_path / zip_file.stem
            futures.append(executor.submit(extract_zip, zip_file, extract_dir))
            
        for future in as_completed(futures):
            future.result()

    # 2. Copy UGV-NBWASTE images
    ugv_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        ugv_images.extend(ugv_raw.rglob(ext))
        
    print(f"Found {len(ugv_images)} images in UGV-NBWASTE. Copying...")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for img_path in ugv_images:
            # Maintain train/valid/test structure roughly
            rel_parts = img_path.relative_to(ugv_raw).parts
            # simplify by just taking the last 2 parts e.g. test/images/img.jpg
            if len(rel_parts) >= 2:
                sub_dir = Path(*rel_parts[-2:-1])
            else:
                sub_dir = Path("misc")
            dst_dir = target_dir / "UGV" / sub_dir
            futures.append(executor.submit(copy_file, img_path, dst_dir))
            
        for i, future in enumerate(as_completed(futures)):
            if i % 500 == 0 and i > 0:
                print(f"Copied {i} UGV images...")

    print("Dataset setup complete! All images are now in dataset/raw/")

if __name__ == "__main__":
    main()
