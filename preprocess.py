import cv2
import numpy as np


def preprocess(image_path):
    """Standard preprocessing pipeline — resize, color convert, denoise, equalize."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, (256, 256))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Noise reduction — slightly stronger kernel helps Alba's thin quills
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # CLAHE instead of plain equalizeHist — better local contrast for thin quills
    # This improves segmentation quality on Alba (thinner, lighter quills)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(blurred)

    return img, gray, hsv, equalized


# ── Augmentation helpers (used by classify.py for Alba oversampling) ──────────

def augment_image_to_features(image_path, extract_fn, n_augments=5):
    """
    Generate n_augments extra feature vectors from one image using
    realistic augmentations. Used to oversample the Alba class.

    Args:
        image_path  : path to the original image
        extract_fn  : features.extract_all — passed in to avoid circular import
        n_augments  : how many extra samples to generate

    Returns:
        list of feature vectors (each is the same length as extract_fn output)
    """
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        return []

    img_orig = cv2.resize(img_orig, (256, 256))
    extras = []

    rng = np.random.default_rng(seed=abs(hash(image_path)) % (2**32))

    for _ in range(n_augments):
        aug = img_orig.copy()

        # 1. Random horizontal flip
        if rng.random() > 0.5:
            aug = cv2.flip(aug, 1)

        # 2. Random rotation ±15°
        angle = rng.uniform(-15, 15)
        h, w  = aug.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        aug   = cv2.warpAffine(aug, M, (w, h),
                               borderMode=cv2.BORDER_REFLECT_101)

        # 3. Brightness jitter ±20%
        factor = rng.uniform(0.80, 1.20)
        aug    = np.clip(aug.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 4. Slight Gaussian noise (mimics real scan variation)
        noise = rng.normal(0, 4, aug.shape).astype(np.int16)
        aug   = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 5. Save to a temp file so extract_all can call preprocess normally
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, aug)
        tmp.close()

        try:
            feats = extract_fn(tmp.name)
            extras.append(feats)
        except Exception:
            pass
        finally:
            os.unlink(tmp.name)

    return extras


def test_preprocess():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    # Find any test image automatically
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
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    titles = ["Original", "Grayscale", "HSV (H channel)", "CLAHE equalized"]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray, hsv[:, :, 0], eq]
    for ax, title, im in zip(axes, titles, images):
        ax.imshow(im, cmap='gray' if len(im.shape) == 2 else None)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/preprocess_result.png", dpi=150)
    plt.close()
    print("outputs/preprocess_result.png saved")


if __name__ == "__main__":
    test_preprocess()
