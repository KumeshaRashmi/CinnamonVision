import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ── Geometric features ─────────────────────────────────────────────────────────
def extract_geometric(contour, mask):
    if contour is None:
        return [0] * 7
    area       = cv2.contourArea(contour)
    perimeter  = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    extent       = area / (w * h) if (w * h) > 0 else 0
    circularity  = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    hull         = cv2.convexHull(contour)
    hull_area    = cv2.contourArea(hull)
    solidity     = area / hull_area if hull_area > 0 else 0
    return [area, perimeter, aspect_ratio, extent, circularity, solidity, float(w)]

# ── Color features (HSV) ───────────────────────────────────────────────────────
def extract_color(hsv_img, mask):
    masked = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
    feats = []
    for ch in range(3):
        channel = masked[:, :, ch]
        pixels  = channel[mask > 0]
        if len(pixels) == 0:
            feats += [0, 0, 0, 0]
            continue
        feats += [float(np.mean(pixels)),
                  float(np.std(pixels)),
                  float(np.min(pixels)),
                  float(np.max(pixels))]
    return feats  # 12 values

# ── Texture features (GLCM + LBP) ─────────────────────────────────────────────
def extract_texture(gray_img, mask):
    roi = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    
    # GLCM
    glcm = graycomatrix(roi, distances=[1], angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy      = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    # LBP
    lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    
    glcm_feats = [contrast, homogeneity, energy, correlation]
    return glcm_feats + list(lbp_hist)  # 14 values

# ── Combined feature vector ────────────────────────────────────────────────────
def extract_all(image_path):
    from preprocess import preprocess
    from segment import segment
    
    img, gray, hsv, eq = preprocess(image_path)
    mask, contour      = segment(eq)
    
    geo     = extract_geometric(contour, mask)
    color   = extract_color(hsv, mask)
    texture = extract_texture(gray, mask)
    
    return geo + color + texture  # 7 + 12 + 14 = 33 features