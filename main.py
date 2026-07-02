import os
from classify import train_and_save
from evaluate import evaluate, visualize_sample_predictions

os.makedirs("outputs", exist_ok=True)

print("=" * 55)
print("  CinnamonVision — Automated Cinnamon Grade Classifier")
print("=" * 55)

print("\n[Step 1/3]  Training with overfitting controls...")
print("  - Alba oversampling (5x augmentation)")
print("  - SelectKBest feature selection (top 20 of 33)")
print("  - GridSearchCV for SVM C/gamma and KNN k")
print("  - class_weight='balanced' to protect Alba class")
print("  - 80/20 stratified train-test split\n")

model, X, y = train_and_save(data_dir="data")

print("\n[Step 2/3]  Generating evaluation plots...")
evaluate()

print("\n[Step 3/3]  Generating sample prediction images...")
visualize_sample_predictions(data_dir="data")

print("\n" + "=" * 55)
print("  Done — all outputs saved to outputs/")
print("=" * 55)
print("  outputs/confusion_matrix.png")
print("  outputs/feature_importance.png")
print("  outputs/per_class_accuracy.png   ← Alba in red")
print("  outputs/alba_analysis.png        ← Alba P/R/F1 detail")
print("  outputs/sample_predictions.png")
print("  outputs/features.csv")
print("  outputs/model.pkl")
