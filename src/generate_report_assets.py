import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import contextlib
from pathlib import Path

# Silence OpenCV logs
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

@contextlib.contextmanager
def suppress_c_stderr():
    """Suppresses C-level stderr output like libjpeg warnings."""
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

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#d1d5db'
plt.rcParams['axes.linewidth'] = 0.8

def load_and_resize(path, max_dim=800):
    with suppress_c_stderr():
        img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def extract_canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def compute_metrics(orig, target):
    """Computes exact PSNR and standardized SSIM for display."""
    from skimage.metrics import structural_similarity as ssim
    if orig.shape != target.shape:
        target = cv2.resize(target, (orig.shape[1], orig.shape[0]))
    mse = np.mean((orig.astype(float) - target.astype(float))**2)
    p = 10 * np.log10(255**2 / mse) if mse > 0 else 100.0
    g1 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    scale = 640 / max(g1.shape)
    g1s = cv2.resize(g1, (int(g1.shape[1]*scale), int(g1.shape[0]*scale)))
    g2s = cv2.resize(g2, (int(g2.shape[1]*scale), int(g2.shape[0]*scale)))
    s = ssim(g1s, g2s, data_range=255)
    return p, s

def generate_diverse_comparisons():
    output_dir = Path("docs/assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("dataset/raw")
    deg_dir = Path("data/degraded")
    rest_dir = Path("data/restored")

    # Define diverse waste items across different material categories
    showcase_items = [
        {
            "category": "Plastic Bottle (Plastic / Recyclable)",
            "flaw_name": "Defocus Blur",
            "filter_name": "Unsharp Masking",
            "flaw_type": "blur",
            "rel_path": Path("BDWaste/Indigestive/Bottle/9. Bottle/1.jpg")
        },
        {
            "category": "Snack Foil Pack (Indigestible Packaging)",
            "flaw_name": "Underexposure (Gamma > 1)",
            "filter_name": "CLAHE (8x8 Grid)",
            "flaw_type": "dark",
            "rel_path": Path("BDWaste/Indigestive/Chips  packet/8. Chips  packet/1.jpg")
        },
        {
            "category": "Banana Peel (Organic Food Waste)",
            "flaw_name": "Additive Sensor Noise",
            "filter_name": "Median Filter (5x5)",
            "flaw_type": "noise",
            "rel_path": Path("BDWaste/Digestive/Banana peel/9. Banana peel/1.jpg")
        }
    ]

    # Additional slide case
    additional_items = [
        {
            "category": "Aluminum Beverage Can (Metal Recyclable)",
            "flaw_name": "Defocus Blur",
            "filter_name": "Unsharp Masking",
            "flaw_type": "blur",
            "rel_path": Path("BDWaste/Indigestive/Cane/2.Cane/1.jpg"),
            "slide_filename": "slide_can_case_study.png"
        }
    ]

    # 1. Master 3x5 Grid Figure for Reports (Diverse Trash)
    fig, axes = plt.subplots(3, 5, figsize=(17, 10.5), dpi=300)
    plt.subplots_adjust(wspace=0.06, hspace=0.26)

    col_titles = [
        "(a) Clean Ground Truth", 
        "(b) Degraded Flaw", 
        "(c) Restored Output", 
        "(d) Degraded Canny Edges", 
        "(e) Restored Canny Edges"
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, pad=10, fontweight='bold', color='#111827')

    for row_idx, item in enumerate(showcase_items):
        flaw = item["flaw_type"]
        rel = item["rel_path"]
        orig_p = raw_dir / rel
        deg_p = deg_dir / rel.parent / f"{rel.stem}_{flaw}{rel.suffix}"
        rest_p = rest_dir / rel.parent / f"{rel.stem}_{flaw}{rel.suffix}"

        orig_bgr = load_and_resize(orig_p)
        deg_bgr = load_and_resize(deg_p)
        rest_bgr = load_and_resize(rest_p)

        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB)
        rest_rgb = cv2.cvtColor(rest_bgr, cv2.COLOR_BGR2RGB)

        deg_canny = extract_canny(deg_bgr)
        rest_canny = extract_canny(rest_bgr)

        # Compute exact metrics
        dp, ds = compute_metrics(orig_bgr, deg_bgr)
        rp, rs = compute_metrics(orig_bgr, rest_bgr)

        # Col 0: Original
        axes[row_idx, 0].imshow(orig_rgb)
        label_y = f"{item['category']}\n[{item['flaw_name']} -> {item['filter_name']}]"
        axes[row_idx, 0].set_ylabel(label_y, fontsize=10, fontweight='bold', color='#1f2937', labelpad=8)

        # Col 1: Degraded
        axes[row_idx, 1].imshow(deg_rgb)
        axes[row_idx, 1].set_xlabel(f"PSNR: {dp:.2f} dB\nSSIM: {ds:.3f}", fontsize=10, color='#b91c1c', fontweight='bold')

        # Col 2: Restored
        axes[row_idx, 2].imshow(rest_rgb)
        axes[row_idx, 2].set_xlabel(f"PSNR: {rp:.2f} dB\nSSIM: {rs:.3f}", fontsize=10, color='#15803d', fontweight='bold')

        # Col 3: Degraded Canny
        axes[row_idx, 3].imshow(deg_canny, cmap='gray')
        axes[row_idx, 3].set_xlabel("Distorted / Missing Edges", fontsize=9.5, color='#b91c1c')

        # Col 4: Restored Canny
        axes[row_idx, 4].imshow(rest_canny, cmap='gray')
        axes[row_idx, 4].set_xlabel("Recovered Contour", fontsize=9.5, color='#15803d')

        for c in range(5):
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])

    master_path = output_dir / "figure1_master_pipeline_comparison.png"
    plt.savefig(master_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated diverse master figure: {master_path}")

    # 2. Individual Slide Figures (Widescreen 16:9 for presentations)
    all_slide_items = showcase_items + additional_items
    for item in all_slide_items:
        flaw = item["flaw_type"]
        rel = item["rel_path"]
        orig_p = raw_dir / rel
        deg_p = deg_dir / rel.parent / f"{rel.stem}_{flaw}{rel.suffix}"
        rest_p = rest_dir / rel.parent / f"{rel.stem}_{flaw}{rel.suffix}"

        orig_bgr = load_and_resize(orig_p)
        deg_bgr = load_and_resize(deg_p)
        rest_bgr = load_and_resize(rest_p)

        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB)
        rest_rgb = cv2.cvtColor(rest_bgr, cv2.COLOR_BGR2RGB)

        rest_canny = extract_canny(rest_bgr)
        dp, ds = compute_metrics(orig_bgr, deg_bgr)
        rp, rs = compute_metrics(orig_bgr, rest_bgr)

        fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.5), dpi=300)
        fig.suptitle(f"Case Study: {item['category']} | {item['flaw_name']} -> {item['filter_name']}", fontsize=13, fontweight='bold', y=0.98)

        axs[0].imshow(orig_rgb)
        axs[0].set_title("(1) Clean Ground Truth", fontsize=10.5, fontweight='bold')

        axs[1].imshow(deg_rgb)
        axs[1].set_title(f"(2) Degraded ({flaw.capitalize()})", fontsize=10.5, fontweight='bold')
        axs[1].set_xlabel(f"PSNR: {dp:.2f} dB | SSIM: {ds:.3f}", color='#b91c1c', fontweight='bold', fontsize=9.5)

        axs[2].imshow(rest_rgb)
        axs[2].set_title(f"(3) Restored ({item['filter_name']})", fontsize=10.5, fontweight='bold')
        axs[2].set_xlabel(f"PSNR: {rp:.2f} dB | SSIM: {rs:.3f}", color='#15803d', fontweight='bold', fontsize=9.5)

        axs[3].imshow(rest_canny, cmap='gray')
        axs[3].set_title("(4) Restored Canny Edges", fontsize=10.5, fontweight='bold')
        axs[3].set_xlabel("Clean Silhouette Recovery", color='#15803d', fontsize=9.5)

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        slide_file = item.get("slide_filename", f"slide_{flaw}_case_study.png")
        slide_path = output_dir / slide_file
        plt.savefig(slide_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated slide: {slide_path}")

def generate_benchmark_charts():
    output_dir = Path("docs/assets")
    csv_path = Path("data/evaluation_metrics.csv")
    if not csv_path.exists():
        print("No evaluation CSV found.")
        return
        
    df = pd.read_csv(csv_path)
    
    grouped = df.groupby("flaw_type").agg({
        "deg_psnr": "mean",
        "rest_psnr": "mean",
        "deg_ssim": "mean",
        "rest_ssim": "mean",
        "name": "count"
    }).reindex(["blur", "dark", "noise"])
    
    flaws = ["Defocus Blur\n(Unsharp Mask)", "Underexposure\n(CLAHE)", "Sensor Noise\n(Median Filter)"]
    x = np.arange(len(flaws))
    width = 0.32

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle(f"Quantitative Restoration Benchmark Across Full Dataset (N = {len(df):,} Images)", fontsize=13, fontweight='bold', y=0.98)

    # 1. PSNR Bar Chart
    rects1 = ax1.bar(x - width/2, grouped["deg_psnr"], width, label="Degraded Input", color="#ef4444", alpha=0.88, edgecolor="#991b1b")
    rects2 = ax1.bar(x + width/2, grouped["rest_psnr"], width, label="Restored Output", color="#10b981", alpha=0.88, edgecolor="#065f46")
    ax1.set_ylabel("Peak Signal-to-Noise Ratio (PSNR in dB)", fontsize=10, fontweight='bold')
    ax1.set_title("PSNR Improvement (Higher is Better)", fontsize=11, fontweight='bold', pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(flaws, fontsize=9.5)
    ax1.legend(loc="upper left", frameon=True, facecolor="white")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, max(grouped["rest_psnr"]) * 1.25)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#065f46")

    # 2. SSIM Bar Chart
    rects3 = ax2.bar(x - width/2, grouped["deg_ssim"], width, label="Degraded Input", color="#ef4444", alpha=0.88, edgecolor="#991b1b")
    rects4 = ax2.bar(x + width/2, grouped["rest_ssim"], width, label="Restored Output", color="#3b82f6", alpha=0.88, edgecolor="#1e40af")
    ax2.set_ylabel("Structural Similarity Index (SSIM: 0 to 1)", fontsize=10, fontweight='bold')
    ax2.set_title("Structural Similarity Index (SSIM)", fontsize=11, fontweight='bold', pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(flaws, fontsize=9.5)
    ax2.legend(loc="upper left", frameon=True, facecolor="white")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1.15)

    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#1e40af")

    chart_path = output_dir / "figure2_quantitative_benchmarks.png"
    plt.savefig(chart_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {chart_path}")

if __name__ == "__main__":
    generate_diverse_comparisons()
    generate_benchmark_charts()
