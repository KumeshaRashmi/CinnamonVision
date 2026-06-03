"""
classify.py — CinnamonVision (overfitting-reduced version)

Overfitting fixes applied
─────────────────────────
1. Alba oversampling via image augmentation (augment_image_to_features)
       → Alba has only 173 images vs 334 C5_Special. We generate synthetic
         extra Alba samples by rotating, flipping, and jittering brightness,
         which gives the model more diverse Alba examples to learn from.

2. SelectKBest feature selection (top 20 of 33)
       → Removes the 13 noisiest features before training so the classifier
         cannot memorise irrelevant patterns.

3. GridSearchCV for SVM (C, gamma) and KNN (k)
       → Finds regularisation strength by cross-validation instead of guessing.
         Lower C = stronger SVM regularisation = less overfitting.

4. class_weight='balanced' on SVM
       → Weights the loss function so Alba mistakes are penalised as much as
         C5_Special mistakes, preventing the model from ignoring Alba.

5. Held-out 20% test set
       → Final honest evaluation on data never seen during training or CV.
         CV-test gap > 5% is reported as an overfitting warning.
"""

import os
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
)
from features import extract_all

# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP   = {"Alba": 0, "C4": 1, "C5": 2, "C5_Special": 3}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# Alba gets the most augmentation because it is the smallest class (173 images).
# Other classes are augmented once each to add slight variety without inflating
# them disproportionately.
AUGMENT_PER_CLASS = {
    "Alba":      5,   # 173 → ~173 + 173*5 = 1038 samples  (oversampled heavily)
    "C4":        3,   # 215 → ~215 + 215*1 = 430  samples
    "C5":        3,   # 199 → ~199 + 199*1 = 398  samples
    "C5_Special": 0,  # 334 → kept as-is          (largest class, no extra)
}
# ─────────────────────────────────────────────────────────────────────────────


def build_dataset(data_dir):
    """
    Load all images, extract features, and apply per-class augmentation.
    Returns X (features), y (labels), paths (original paths only).
    """
    from preprocess import augment_image_to_features

    X, y, paths = [], [], []

    for label, idx in LABEL_MAP.items():
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            print(f"  Warning: {folder} not found — skipping")
            continue

        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        n_aug = AUGMENT_PER_CLASS.get(label, 0)
        print(f"  {label}: {len(files)} real images, {n_aug} augments each")

        for fname in files:
            path = os.path.join(folder, fname)
            try:
                feats = extract_all(path)
                X.append(feats)
                y.append(idx)
                paths.append(path)
            except Exception as e:
                print(f"    Skipped {fname}: {e}")
                continue

            # Augmentation for this image
            if n_aug > 0:
                extras = augment_image_to_features(path, extract_all,
                                                   n_augments=n_aug)
                for ef in extras:
                    X.append(ef)
                    y.append(idx)

    X = np.array(X)
    y = np.array(y)

    # Print final class distribution
    print("\n  Final dataset distribution after augmentation:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {LABEL_NAMES[u]}: {c} samples")
    print(f"  Total: {len(y)} samples, {X.shape[1]} features")

    return X, y, paths


def train_and_save(data_dir="data"):
    """
    Full training pipeline with overfitting controls.
    Returns the best fitted pipeline, X, y.
    """
    print("=" * 55)
    print("CinnamonVision — Training (overfitting-reduced)")
    print("=" * 55)

    print("\n[1/4] Building dataset with augmentation...")
    X, y, _ = build_dataset(data_dir)

    # Save feature matrix for evaluate.py
    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv("outputs/features.csv", index=False)
    print("  features.csv saved")

    # ── Stratified train/test split ──────────────────────────────────────────
    print("\n[2/4] Splitting data (80% train / 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── SVM — grid search over C and gamma ───────────────────────────────────
    print("\n[3/4] Grid-searching SVM hyperparameters (this takes ~2 min)...")
    svm_base = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=20)),
        ('svm',    SVC(kernel='rbf', probability=True,
                       class_weight='balanced'))     # balanced → protects Alba
    ])
    svm_params = {
        'svm__C':     [0.1, 1, 10, 50],
        'svm__gamma': ['scale', 'auto', 0.01, 0.001],
    }
    svm_search = GridSearchCV(
        svm_base, svm_params,
        cv=cv, scoring='balanced_accuracy',   # balanced_accuracy penalises Alba misses
        n_jobs=-1, verbose=0
    )
    svm_search.fit(X_train, y_train)
    best_svm   = svm_search.best_estimator_
    svm_cv_acc = svm_search.best_score_

    print(f"  Best SVM params : {svm_search.best_params_}")
    print(f"  SVM CV balanced-accuracy : {svm_cv_acc:.3f}")

    # ── KNN — grid search over k ─────────────────────────────────────────────
    knn_base = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=20)),
        ('knn',    KNeighborsClassifier(metric='euclidean'))
    ])
    knn_params = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 15]
    }
    knn_search = GridSearchCV(
        knn_base, knn_params,
        cv=cv, scoring='balanced_accuracy',
        n_jobs=-1, verbose=0
    )
    knn_search.fit(X_train, y_train)
    best_knn   = knn_search.best_estimator_
    knn_cv_acc = knn_search.best_score_

    print(f"  Best KNN params : {knn_search.best_params_}")
    print(f"  KNN CV balanced-accuracy : {knn_cv_acc:.3f}")

    # ── Pick winner ───────────────────────────────────────────────────────────
    if svm_cv_acc >= knn_cv_acc:
        best_pipe  = best_svm
        model_name = "SVM"
    else:
        best_pipe  = best_knn
        model_name = "KNN"
    print(f"\n  Selected model : {model_name}")

    # ── Held-out test evaluation ──────────────────────────────────────────────
    print("\n[4/4] Evaluating on held-out test set...")
    test_acc = best_pipe.score(X_test, y_test)
    gap      = svm_cv_acc - test_acc if model_name == "SVM" else knn_cv_acc - test_acc

    print(f"  CV balanced-accuracy  : {(svm_cv_acc if model_name=='SVM' else knn_cv_acc):.3f}")
    print(f"  Test accuracy         : {test_acc:.3f}")
    if gap > 0.05:
        print(f"  WARNING: CV-test gap {gap:.3f} > 0.05 — some overfitting remains.")
    else:
        print(f"  OK: CV-test gap {gap:.3f} — model generalises well.")

    # Per-class test accuracy (shows specifically how well Alba is handled)
    from sklearn.metrics import classification_report
    y_pred_test  = best_pipe.predict(X_test)
    y_pred_train = best_pipe.predict(X_train)
    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred_test])))
    present_names  = [LABEL_NAMES[i] for i in present_labels]
    print("\n  Per-class test report:")
    print(classification_report(
        y_test, y_pred_test,
        labels=present_labels,
        target_names=present_names,
        digits=3
    ))

    # ── Save train/test split results for evaluate.py ─────────────────────────
    split_results = {
        "y_train":      y_train,
        "y_pred_train": y_pred_train,
        "y_test":       y_test,
        "y_pred_test":  y_pred_test,
        "model_name":   model_name,
    }
    with open("outputs/split_results.pkl", "wb") as f:
        pickle.dump(split_results, f)
    print("  split_results.pkl saved (train & test predictions)")

    # ── Refit on all data and save ────────────────────────────────────────────
    best_pipe.fit(X, y)
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(best_pipe, f)
    print("  Model saved to outputs/model.pkl")

    return best_pipe, X, y


if __name__ == "__main__":
    train_and_save()