import sys
import os
import json
import argparse
import pickle
import numpy as np

LABEL_NAMES = {0: "Alba", 1: "C4", 2: "C5", 3: "C5_Special"}

GRADE_INFO = {
    "Alba": {
        "rank": 1,
        "description": "Highest quality grade. Thin, tightly rolled pale quills with delicate flavour.",
        "color": "#7F77DD",
        "uses": "Fine dining, premium baking, direct consumption"
    },
    "C4": {
        "rank": 2,
        "description": "Premium grade. Thicker quills, strong aroma, widely used commercially.",
        "color": "#1D9E75",
        "uses": "Culinary, beverages, confectionery"
    },
    "C5": {
        "rank": 3,
        "description": "Standard grade. Coarser texture, bold flavour, cost-effective.",
        "color": "#EF9F27",
        "uses": "Spice blends, food manufacturing, everyday cooking"
    },
    "C5_Special": {
        "rank": 4,
        "description": "Economy grade. Rough cut, strong and pungent, used in bulk applications.",
        "color": "#D85A30",
        "uses": "Industrial food processing, extracts, bulk spice"
    },
}


def predict_image(image_path, model_path="outputs/model.pkl"):
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    if not os.path.exists(model_path):
        return {"error": f"Model not found: {model_path}. Run main.py first."}

    try:
        from features import extract_all
        feats = extract_all(image_path)
    except Exception as e:
        return {"error": f"Feature extraction failed: {e}"}

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return {"error": f"Model load failed: {e}"}

    feats_arr = np.array([feats])
    pred_idx  = model.predict(feats_arr)[0]
    pred_name = LABEL_NAMES.get(pred_idx, f"Class_{pred_idx}")

    probs = {}
    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(feats_arr)[0]
        # Map to class names — use model's classes_ to be safe
        classes = model.classes_ if hasattr(model, "classes_") else list(range(len(prob_arr)))
        for cls_idx, prob in zip(classes, prob_arr):
            name = LABEL_NAMES.get(cls_idx, f"Class_{cls_idx}")
            probs[name] = round(float(prob), 4)
    else:
        probs[pred_name] = 1.0

    confidence = probs.get(pred_name, 1.0)

    return {
        "prediction":   pred_name,
        "confidence":   confidence,
        "probabilities": probs,
        "grade_info":   GRADE_INFO.get(pred_name, {}),
        "image_path":   image_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CinnamonVision single-image predictor")
    parser.add_argument("image", help="Path to cinnamon image")
    parser.add_argument("--model", default="outputs/model.pkl", help="Path to model.pkl")
    args = parser.parse_args()

    result = predict_image(args.image, args.model)
    print(json.dumps(result, indent=2))