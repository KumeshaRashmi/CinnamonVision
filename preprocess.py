import cv2
import numpy as np

def preprocess(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Contrast enhancement
    equalized = cv2.equalizeHist(blurred)
    
    return img, gray, hsv, equalized

def test_preprocess():
    import matplotlib.pyplot as plt
    img, gray, hsv, eq = preprocess("data/Alba/sample.jpg")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    titles = ["Original", "Grayscale", "HSV (H channel)", "Equalized"]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gray, hsv[:,:,0], eq]
    for ax, title, im in zip(axes, titles, images):
        ax.imshow(im, cmap='gray' if len(im.shape)==2 else None)
        ax.set_title(title); ax.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/preprocess_result.png")
    plt.show()

if __name__ == "__main__":
    test_preprocess()