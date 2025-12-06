import sys
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np

# Project root (folder where pipeline.py is located)
ROOT = Path(__file__).parent

# Input and output folders
INPUT_DIR = ROOT / "input_bw"
DDCOLOR_RAW_DIR = ROOT / "output_color_raw"
GFPGAN_OUTPUT_DIR = ROOT / "output_faces_restored"
FINAL_DIR = ROOT / "output_final"

# Third party repositories
DDCOLOR_DIR = ROOT / "third_party" / "DDColor"
GFPGAN_DIR = ROOT / "third_party" / "GFPGAN"

# DDColor model name
MODEL_NAME = "ddcolor_paper"  # or "ddcolor_paper"


def run_ddcolor():
    """Run DDColor on black and white photos and copy results to output_color_raw."""
    print("=== Step 1: DDColor colorization ===")

    # Ensure output_color_raw exists and is empty
    DDCOLOR_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for p in DDCOLOR_RAW_DIR.glob("*"):
        if p.is_file():
            p.unlink()

    # Clean DDColor internal results folder
    ddcolor_results_dir = DDCOLOR_DIR / "results"
    if ddcolor_results_dir.exists():
        for p in ddcolor_results_dir.glob("*"):
            if p.is_file():
                p.unlink()

    # Run DDColor inference script
    cmd = [
        sys.executable,
        "infer_hf.py",
        "--model_name",
        MODEL_NAME,
        "--input",
        str(INPUT_DIR),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(DDCOLOR_DIR), check=True)

    # Copy DDColor results into project-level output_color_raw
    if ddcolor_results_dir.exists():
        for p in ddcolor_results_dir.glob("*.*"):
            if p.is_file():
                shutil.copy2(p, DDCOLOR_RAW_DIR / p.name)

    print("DDColor done. Raw color images in:", DDCOLOR_RAW_DIR)


def run_gfpgan():
    """Run GFPGAN on DDColor outputs to restore faces."""
    print("=== Step 2: GFPGAN face restoration ===")

    GFPGAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "inference_gfpgan.py",
        "-i",
        str(DDCOLOR_RAW_DIR),
        "-o",
        str(GFPGAN_OUTPUT_DIR),
        "-v",
        "1.4",
        "-s",
        "1",  # upscale x1 (no resize)
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(GFPGAN_DIR), check=True)

    print("GFPGAN done. Results in:", GFPGAN_OUTPUT_DIR)


def enhance_image(img: np.ndarray) -> np.ndarray:
    """
    Soft postprocessing for restored photos:
    - remove hot pixels in dark areas
    - reduce oversaturation
    - soften magenta/red overshoot
    - fix gray/cold skin patches (ears, neck, hands)
    - add very light film grain
    """

    result = img.copy()

    # ---------------------------------
    # 1. Remove hot pixels in dark areas
    # ---------------------------------
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    dark_mask = gray < 40
    median = cv2.medianBlur(result, 3)

    for c in range(3):
        result[:, :, c][dark_mask] = median[:, :, c][dark_mask]

    # ---------------------------------
    # 2. Slightly reduce overall saturation
    # ---------------------------------
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.85
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ---------------------------------
    # 3. Fix gray/cold skin patches
    # ---------------------------------
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Skin detection in LAB
    skin_mask = (
        (A > 125) & (A < 160) &
        (B > 120) & (B < 170) &
        (L > 40)
    )

    if np.any(skin_mask):
        skin_A_mean = np.mean(A[skin_mask])
        skin_B_mean = np.mean(B[skin_mask])

        # Gray/cold skin = low A/B
        gray_skin_mask = skin_mask & ((A < skin_A_mean - 6) | (B < skin_B_mean - 6))

        # Correct toward average skin tone
        A[gray_skin_mask] = (A[gray_skin_mask] * 0.5 + skin_A_mean * 0.5).astype(np.uint8)
        B[gray_skin_mask] = (B[gray_skin_mask] * 0.5 + skin_B_mean * 0.5).astype(np.uint8)

    lab_fixed = cv2.merge([L, A, B])
    result = cv2.cvtColor(lab_fixed, cv2.COLOR_LAB2BGR)

    # ---------------------------------
    # 4. Soften red/magenta overshoot
    # ---------------------------------
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    A = cv2.addWeighted(A, 0.9, np.full_like(A, 128), 0.1, 0)
    result = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    # ---------------------------------
    # 5. Very light film grain
    # ---------------------------------
    noise_sigma = 2.0
    noise = np.random.normal(0, noise_sigma, result.shape).astype(np.float32)
    result_float = result.astype(np.float32) + noise
    result = np.clip(result_float, 0, 255).astype(np.uint8)

    return result


def postprocess_results():
    """Apply final polishing to GFPGAN output images."""
    print("=== Step 3: Final tone & color adjustment ===")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # GFPGAN usually saves restored faces into a subfolder called "restored_imgs"
    restored_dir = GFPGAN_OUTPUT_DIR / "restored_imgs"
    source_dir = restored_dir if restored_dir.exists() else GFPGAN_OUTPUT_DIR

    if not any(source_dir.glob("*.*")):
        print("No images found to postprocess in:", source_dir)
        return

    for img_path in source_dir.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None:
            print("Skipping unreadable:", img_path)
            continue

        enhanced = enhance_image(img)
        out_path = FINAL_DIR / img_path.name
        cv2.imwrite(str(out_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        print("Saved:", out_path)

    print("Done. Final results in:", FINAL_DIR)


if __name__ == "__main__":
    run_ddcolor()
    run_gfpgan()
    postprocess_results()
