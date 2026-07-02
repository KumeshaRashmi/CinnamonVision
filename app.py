# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile, os
from predict import predict_image

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    suffix = os.path.splitext(file.filename)[1] or ".jpg"

    # On Windows, delete=False + manual unlink is required
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    try:
        file.save(tmp_path)
        tmp.close()                  # ← close before reading/deleting
        result = predict_image(tmp_path)
    finally:
        tmp.close()
        os.unlink(tmp_path)          # ← now safe to delete

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)