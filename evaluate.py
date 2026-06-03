"""
evaluate.py — CinnamonVision (overfitting-reduced version)

Changes from original
─────────────────────
- CLASS_NAMES fixed to include all 4 classes (was missing C5_Special).
- Per-class accuracy plot now highlights Alba in red so underfitting is visible.
- Added Alba recall trend printout after the classification report.
- confusion_matrix saved with correct 4-class labels.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# ── Must match LABEL_MAP in classify.py exactly ──────────────────────────────
CLASS_NAMES = ["Alba", "C4", "C5", "C5_Special"]

FEAT_NAMES = (
    ["area", "perimeter", "aspect_ratio", "extent",
     "circularity", "solidity", "width"] +
    [f"color_{i}" for i in range(12)] +
    ["contrast", "homogeneity", "energy", "correlation"] +
    [f"lbp_{i}" for i in range(10)]
)


def evaluate():
    """
    Load features.csv, model.pkl, and split_results.pkl, then save:
      - confusion_matrix.png       (side-by-side: train | test)
      - feature_importance.png
      - per_class_accuracy.png     (grouped bars: train vs test, Alba highlighted)
      - alba_analysis.png          (Alba P/R/F1 for train vs test)
    """
    import pickle as _pickle

    df = pd.read_csv("outputs/features.csv")
    y  = df["label"].values
    X  = df.drop("label", axis=1).values
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features from features.csv")

    with open("outputs/model.pkl", "rb") as f:
        model = _pickle.load(f)

    # ── Load saved train/test split results ───────────────────────────────────
    split_path = "outputs/split_results.pkl"
    if os.path.exists(split_path):
        with open(split_path, "rb") as f:
            sr = _pickle.load(f)
        y_train      = sr["y_train"]
        y_pred_train = sr["y_pred_train"]
        y_test       = sr["y_test"]
        y_pred_test  = sr["y_pred_test"]
        model_name   = sr.get("model_name", "Model")
        print(f"  Loaded split_results.pkl ({model_name})")
        print(f"  Train samples: {len(y_train)} | Test samples: {len(y_test)}")
    else:
        # Fallback: CV predict on full dataset (old behaviour)
        print("  split_results.pkl not found — falling back to CV predict on full data")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_test  = cross_val_predict(model, X, y, cv=cv)
        y_train      = y
        y_pred_train = y_pred_test
        y_test       = y
        model_name   = "CV"

    present_labels = sorted(np.unique(np.concatenate([y_train, y_test,
                                                       y_pred_train, y_pred_test])))
    target_names   = [CLASS_NAMES[i] for i in present_labels]

    print("\n=== Train Classification Report ===")
    print(classification_report(y_train, y_pred_train, target_names=target_names))
    print("=== Test Classification Report ===")
    print(classification_report(y_test, y_pred_test, target_names=target_names))

    cm_train = confusion_matrix(y_train, y_pred_train, labels=present_labels)
    cm_test  = confusion_matrix(y_test,  y_pred_test,  labels=present_labels)

    # ── 1. Confusion matrix — side by side ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, cm, title in zip(
        axes,
        [cm_train, cm_test],
        [f"Train set ({len(y_train)} samples)", f"Test set ({len(y_test)} samples)"]
    ):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {title}")
    plt.suptitle(f"CinnamonVision — {model_name} Confusion Matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()
    print("confusion_matrix.png saved")

    # ── 2. Feature importance ─────────────────────────────────────────────────
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)
    variances = np.var(X_scaled, axis=0)
    top_idx   = np.argsort(variances)[::-1][:15]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(15), variances[top_idx],
                  color='steelblue', edgecolor='white')
    ax.set_xticks(range(15))
    ax.set_xticklabels(
        [FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"feat_{i}"
         for i in top_idx],
        rotation=45, ha='right', fontsize=10
    )
    ax.set_title("Top 15 most discriminative features (normalised variance)",
                 fontsize=13)
    ax.set_ylabel("Variance (after StandardScaler)")
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("feature_importance.png saved")

    # ── 3. Per-class accuracy — grouped bars: Train vs Test ──────────────────
    acc_train = cm_train.diagonal() / cm_train.sum(axis=1) * 100
    acc_test  = cm_test.diagonal()  / cm_test.sum(axis=1)  * 100

    x     = np.arange(len(target_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_train = ax.bar(x - width / 2, acc_train, width,
                        label='Train',
                        color=['#EF9A9A' if n == 'Alba' else '#90CAF9'
                               for n in target_names],
                        edgecolor='white')
    bars_test  = ax.bar(x + width / 2, acc_test, width,
                        label='Test',
                        color=['#E53935' if n == 'Alba' else '#1565C0'
                               for n in target_names],
                        edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class Accuracy — Train vs Test\n"
                 "(Alba bars highlighted in red — target ≥ 90%)")
    ax.bar_label(bars_train, fmt='%.1f%%', padding=3, fontsize=9)
    ax.bar_label(bars_test,  fmt='%.1f%%', padding=3, fontsize=9)
    ax.axhline(y=90, color='black', linestyle='--', linewidth=1,
               label='90% target')
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/per_class_accuracy.png", dpi=150)
    plt.close()
    print("per_class_accuracy.png saved")

    # ── 4. Alba precision / recall / F1 — train vs test ──────────────────────
    _plot_alba_analysis(y_train, y_pred_train, y_test, y_pred_test, target_names)

    return y_test, y_pred_test


def _plot_alba_analysis(y_train, y_pred_train, y_test, y_pred_test, target_names):
    """Bar chart showing Precision/Recall/F1 per class — Train vs Test side by side."""
    from sklearn.metrics import precision_recall_fscore_support

    labels = list(range(len(target_names)))
    p_tr, r_tr, f_tr, _ = precision_recall_fscore_support(y_train, y_pred_train, labels=labels)
    p_te, r_te, f_te, _ = precision_recall_fscore_support(y_test,  y_pred_test,  labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, (p, r, f), split in zip(
        axes,
        [(p_tr, r_tr, f_tr), (p_te, r_te, f_te)],
        ["Train", "Test"]
    ):
        x     = np.arange(len(target_names))
        width = 0.25
        alba_p  = '#E53935'; alba_r  = '#B71C1C'; alba_f  = '#FF8F00'
        other_p = '#42A5F5'; other_r = '#1565C0'; other_f = '#26A69A'

        b1 = ax.bar(x - width, p * 100, width, label='Precision',
                    color=[alba_p if n == 'Alba' else other_p for n in target_names],
                    alpha=0.9)
        b2 = ax.bar(x,         r * 100, width, label='Recall',
                    color=[alba_r if n == 'Alba' else other_r for n in target_names],
                    alpha=0.9)
        b3 = ax.bar(x + width, f * 100, width, label='F1',
                    color=[alba_f if n == 'Alba' else other_f for n in target_names],
                    alpha=0.9)

        ax.bar_label(b1, fmt='%.0f%%', padding=2, fontsize=7)
        ax.bar_label(b2, fmt='%.0f%%', padding=2, fontsize=7)
        ax.bar_label(b3, fmt='%.0f%%', padding=2, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(target_names)
        ax.set_ylim(0, 120)
        ax.set_ylabel("Score (%)")
        ax.set_title(f"{split} set — Precision / Recall / F1\n"
                     "(Alba in red/dark-red/orange)")
        ax.axhline(y=90, color='black', linestyle='--', linewidth=0.8,
                   label='90% target')
        ax.legend(fontsize=8)

    plt.suptitle("CinnamonVision — Alba Analysis: Train vs Test", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/alba_analysis.png", dpi=150)
    plt.close()
    print("alba_analysis.png saved")

    # Print Alba numbers for both splits
    alba_idx = list(target_names).index('Alba') if 'Alba' in target_names else None
    if alba_idx is not None:
        print(f"\n  Alba (Train) — P: {p_tr[alba_idx]*100:.1f}%  R: {r_tr[alba_idx]*100:.1f}%  F1: {f_tr[alba_idx]*100:.1f}%")
        print(f"  Alba (Test)  — P: {p_te[alba_idx]*100:.1f}%  R: {r_te[alba_idx]*100:.1f}%  F1: {f_te[alba_idx]*100:.1f}%")


def visualize_sample_predictions(data_dir="data"):
    """Show 8 random sample predictions with true vs predicted grade."""
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
            mask, contour       = segment(eq)
            feats               = extract_all(path)
            pred                = model.predict([feats])[0]

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

    plt.suptitle("Sample predictions — green = correct, red = incorrect",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/sample_predictions.png", dpi=150)
    plt.close()
    print("sample_predictions.png saved")


if __name__ == "__main__":
    evaluate()
    visualize_sample_predictions()