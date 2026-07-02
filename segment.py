import cv2
import numpy as np


def segment(gray):
    """
    Segment the cinnamon quill from the background.

    Improvements over original:
    - Tries Otsu first; falls back to adaptive thresholding if the binary mask
      looks empty — this handles Alba's lighter-coloured, thinner quills which
      sometimes fool pure Otsu.
    - Returns the full-image mask as fallback so feature extraction never crashes.
    """
    h, w = gray.shape

    # ── Attempt 1: Otsu thresholding ─────────────────────────────────────────
    _, binary_otsu = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Attempt 2: Adaptive thresholding (better for uneven lighting) ────────
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=5
    )

    # Choose whichever binary mask has more foreground pixels (more signal)
    coverage_otsu     = np.count_nonzero(binary_otsu)
    coverage_adaptive = np.count_nonzero(binary_adaptive)

    if coverage_otsu >= coverage_adaptive:
        binary = binary_otsu
    else:
        binary = binary_adaptive

    # ── Morphological cleanup ─────────────────────────────────────────────────
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)

    # ── Contour detection ─────────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out tiny noise contours (< 1% of image area)
    min_area   = 0.01 * h * w
    valid      = [c for c in contours if cv2.contourArea(c) >= min_area]

    if valid:
        largest = max(valid, key=cv2.contourArea)
        mask    = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask, largest

    # Fallback: full-image mask so downstream features don't crash
    fallback_mask = np.ones_like(gray, dtype=np.uint8) * 255
    return fallback_mask, None


def test_segment():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from preprocess import preprocess
    import os

    test_path = None
    for root, dirs, files in os.walk("data"):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_path = os.path.join(root, f)
                break
        if test_path:
            break

    if test_path is None:
        print("No test image found in data/")
        return

    img, gray, hsv, eq = preprocess(test_path)
    mask, contour = segment(eq)

    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Binary mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Contour overlay")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/segment_result.png", dpi=150)
    plt.close()
    print("outputs/segment_result.png saved")


if __name__ == "__main__":
    test_segment()
