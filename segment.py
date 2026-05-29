import cv2
import numpy as np

def segment(gray):
    # Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Keep the largest contour (the quill)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask, largest
    
    return cleaned, None

def test_segment():
    import matplotlib.pyplot as plt
    from preprocess import preprocess
    img, gray, hsv, eq = preprocess("data/Alba/sample.jpg")
    mask, contour = segment(eq)
    
    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original")
    axes[1].imshow(mask, cmap='gray'); axes[1].set_title("Binary mask")
    axes[2].imshow(overlay); axes[2].set_title("Contour overlay")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/segment_result.png")
    plt.show()

if __name__ == "__main__":
    test_segment()
