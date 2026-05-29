import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ← saves files without opening any windows at all
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import pickle, os

CLASS_NAMES = ["Alba", "C4", "C5", "C5 Special"]

FEAT_NAMES = (
    ["area", "perimeter", "aspect_ratio", "extent", "circularity", "solidity", "width"] +
    [f"color_{i}" for i in range(12)] +
    ["contrast", "homogeneity", "energy", "correlation"] +
    [f"lbp_{i}" for i in range(10)]
)

def evaluate():
    # Load from CSV — fast, no image reprocessing
    df = pd.read_csv("outputs/features.csv")
    y  = df["label"].values
    X  = df.drop("label", axis=1).values
    print(f"Loaded {X.shape[0]} samples from features.csv ")

    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES))

    # ── 1. Confusion matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — CinnamonVision")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("confusion_matrix.png saved")

    # ── 2. Feature importance (normalized) ───────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    variances = np.var(X_scaled, axis=0)
    top_idx = np.argsort(variances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(15), variances[top_idx], color='steelblue', edgecolor='white')
    ax.set_xticks(range(15))
    ax.set_xticklabels([FEAT_NAMES[i] for i in top_idx], rotation=45, ha='right', fontsize=10)
    ax.set_title("Top 15 most discriminative features (normalized variance)", fontsize=13)
    ax.set_ylabel("Variance (after StandardScaler)")
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("feature_importance.png saved")

    # ── 3. Per-class accuracy bar chart ───────────────────────────────────────
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, per_class_acc * 100,
                  color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class accuracy — CinnamonVision")
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=11)
    ax.axhline(y=89, color='red', linestyle='--', linewidth=1, label='Overall avg 89%')
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/per_class_accuracy.png", dpi=150)
    plt.close()
    print("per_class_accuracy.png saved")

    return y, y_pred


def visualize_sample_predictions(data_dir="data"):
    import cv2
    import random
    from features import extract_all
    from preprocess import preprocess
    from segment import segment

    with open("outputs/model.pkl", "rb") as f:
        model = pickle.load(f)

    grade_folders = {"Alba": 0, "C4": 1, "C5": 2, "C5_Special": 3}
    grade_names   = {0: "Alba", 1: "C4", 2: "C5", 3: "C5 Special"}

    samples = []
    for label, idx in grade_folders.items():
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)
        for fname in files[:2]:
            samples.append((os.path.join(folder, fname), idx))

    samples = samples[:8]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (path, true_label) in enumerate(samples):
        ax = axes[i // 4][i % 4]
        try:
            img, gray, hsv, eq = preprocess(path)
            mask, contour = segment(eq)
            feats = extract_all(path)
            pred = model.predict([feats])[0]

            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if contour is not None:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)

            ax.imshow(display)
            color = 'green' if pred == true_label else 'red'
            ax.set_title(
                f"True: {grade_names[true_label]}\nPred: {grade_names[pred]}",
                color=color, fontsize=10, fontweight='bold'
            )
        except Exception as e:
            ax.set_title(f"Error: {e}", fontsize=8, color='red')
        ax.axis('off')

    plt.suptitle("Sample predictions — green = correct, red = incorrect", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/sample_predictions.png", dpi=150)
    plt.close()
    print("sample_predictions.png saved")


if __name__ == "__main__":
    evaluate()
    visualize_sample_predictions()