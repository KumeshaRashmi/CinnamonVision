import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import pickle
from features import extract_all

LABEL_MAP = {
    "Alba": 0, "C4": 1, "C5": 2, "C5_Special": 3
}

def build_dataset(data_dir):
    X, y, paths = [], [], []
    for label in LABEL_MAP:
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Processing {label}: {len(files)} images...")
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                feats = extract_all(path)
                X.append(feats)
                y.append(LABEL_MAP[label])
                paths.append(path)
            except Exception as e:
                print(f"  Skipped {fname}: {e}")
    return np.array(X), np.array(y), paths

def train_and_save(data_dir="data"):
    X, y, _ = build_dataset(data_dir)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Save for inspection
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv("outputs/features.csv", index=False)
    
    # KNN pipeline
    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn',    KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
    ])
    
    # SVM pipeline
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    SVC(kernel='rbf', C=10, gamma='scale', probability=True))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    knn_scores = cross_val_score(knn_pipe, X, y, cv=cv, scoring='accuracy')
    svm_scores = cross_val_score(svm_pipe, X, y, cv=cv, scoring='accuracy')
    
    print(f"\nKNN 5-fold accuracy: {knn_scores.mean():.3f} ± {knn_scores.std():.3f}")
    print(f"SVM 5-fold accuracy: {svm_scores.mean():.3f} ± {svm_scores.std():.3f}")
    
    # Train final model on all data (use whichever is better)
    best_pipe = svm_pipe if svm_scores.mean() > knn_scores.mean() else knn_pipe
    best_pipe.fit(X, y)
    
    with open("outputs/model.pkl", "wb") as f:
        pickle.dump(best_pipe, f)
    print("\nModel saved to outputs/model.pkl")
    return best_pipe, X, y

if __name__ == "__main__":
    train_and_save()