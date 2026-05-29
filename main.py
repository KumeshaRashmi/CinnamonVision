from classify import train_and_save
from evaluate import evaluate, visualize_sample_predictions
import os

os.makedirs("outputs", exist_ok=True)

print("=" * 50)
print("CinnamonVision — Automated Cinnamon Grading")
print("=" * 50)

print("\n[1/3] Training model...")
model, X, y = train_and_save(data_dir="data")

print("\n[2/3] Evaluating with 5-fold cross-validation...")
evaluate()  # ← removed data_dir argument

print("\n[3/3] Generating sample predictions...")
visualize_sample_predictions(data_dir="data")

print("\nAll outputs saved to outputs/")
print("   - confusion_matrix.png")
print("   - feature_importance.png")
print("   - per_class_accuracy.png")
print("   - sample_predictions.png")