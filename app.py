from flask import Flask, request, jsonify, render_template
import pickle, os, cv2, numpy as np
from werkzeug.utils import secure_filename
from features import extract_geometric, extract_color, extract_texture
from preprocess import preprocess
from segment import segment

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('static/uploads', exist_ok=True)

with open('outputs/model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

GRADE_INFO = {
    0: {"name":"Alba",       "color":"#4A90D9","description":"Premium grade — finest Ceylon cinnamon. Diameter under 6mm, 45+ quills/kg. Commands the highest export price.","price":"Highest","market":"Premium Export","emoji":"🥇"},
    1: {"name":"C4",         "color":"#3A7D44","description":"Good quality quills with slight surface variations. Diameter 6–10mm. Popular high-value export grade.","price":"High","market":"Quality Export","emoji":"🥈"},
    2: {"name":"C5",         "color":"#D4700A","description":"Standard commercial grade. Diameter 10–16mm. Widely traded in bulk international markets.","price":"Medium","market":"Bulk Trade","emoji":"🥉"},
    3: {"name":"C5 Special", "color":"#C8A84B","description":"Special commercial grade. Larger diameter quills with mixed quality characteristics. Common in domestic markets.","price":"Standard","market":"Commercial","emoji":"📦"},
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    allowed = {'png','jpg','jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.',1)[1].lower() not in allowed:
        return jsonify({'error': 'Only PNG/JPG files allowed'}), 400

    filename  = secure_filename(file.filename)
    filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img, gray, hsv, eq = preprocess(filepath)
        mask, contour      = segment(eq)

        # All three feature groups
        geo     = extract_geometric(contour, mask)
        color   = extract_color(hsv, mask)
        texture = extract_texture(gray, mask)
        all_feats = geo + color + texture

        pred  = MODEL.predict([all_feats])[0]
        proba = MODEL.predict_proba([all_feats])[0] if hasattr(MODEL, 'predict_proba') else None

        # Save contour overlay image
        display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if contour is not None:
            cv2.drawContours(display, [contour], -1, (0,255,100), 3)
        result_path = os.path.join('static/uploads', 'result_' + filename)
        cv2.imwrite(result_path, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

        # Confidence
        confidence = {}
        if proba is not None:
            for i, p in enumerate(proba):
                confidence[GRADE_INFO[i]['name']] = round(float(p)*100, 1)
        else:
            confidence[GRADE_INFO[pred]['name']] = 100.0

        # ── Geometric features ──────────────────────────────────────────────
        geometric_feats = {
            "Area":         {"value": int(geo[0]),       "unit": "px²",  "desc": "Total quill surface area"},
            "Perimeter":    {"value": round(geo[1],1),   "unit": "px",   "desc": "Boundary length"},
            "Aspect Ratio": {"value": round(geo[2],3),   "unit": "",     "desc": "Width / height ratio"},
            "Extent":       {"value": round(geo[3],3),   "unit": "",     "desc": "Area vs bounding box"},
            "Circularity":  {"value": round(geo[4],3),   "unit": "",     "desc": "How round the quill is"},
            "Solidity":     {"value": round(geo[5],3),   "unit": "",     "desc": "Convex hull fill ratio"},
            "Width":        {"value": int(geo[6]),        "unit": "px",   "desc": "Quill diameter (pixels)"},
        }

        # ── Color features (HSV stats) ──────────────────────────────────────
        color_feats = {
            "Hue Mean":        {"value": round(color[0],2),  "unit": "°",  "desc": "Average hue (color tone)"},
            "Hue Std":         {"value": round(color[1],2),  "unit": "",   "desc": "Hue variation (uniformity)"},
            "Saturation Mean": {"value": round(color[4],2),  "unit": "",   "desc": "Color intensity"},
            "Saturation Std":  {"value": round(color[5],2),  "unit": "",   "desc": "Saturation variation"},
            "Value Mean":      {"value": round(color[8],2),  "unit": "",   "desc": "Average brightness"},
            "Value Std":       {"value": round(color[9],2),  "unit": "",   "desc": "Brightness variation"},
        }

        # ── Texture features ────────────────────────────────────────────────
        texture_feats = {
            "GLCM Contrast":    {"value": round(texture[0],4), "unit": "", "desc": "Surface roughness measure"},
            "GLCM Homogeneity": {"value": round(texture[1],4), "unit": "", "desc": "Texture smoothness"},
            "GLCM Energy":      {"value": round(texture[2],4), "unit": "", "desc": "Texture uniformity"},
            "GLCM Correlation": {"value": round(texture[3],4), "unit": "", "desc": "Pixel relationship strength"},
            "LBP Mean":         {"value": round(float(np.mean(texture[4:])),4), "unit": "", "desc": "Avg local binary pattern"},
            "LBP Std":          {"value": round(float(np.std(texture[4:])),4),  "unit": "", "desc": "LBP pattern variation"},
        }

        return jsonify({
            'grade':        GRADE_INFO[pred]['name'],
            'grade_id':     int(pred),
            'color':        GRADE_INFO[pred]['color'],
            'description':  GRADE_INFO[pred]['description'],
            'price':        GRADE_INFO[pred]['price'],
            'market':       GRADE_INFO[pred]['market'],
            'emoji':        GRADE_INFO[pred]['emoji'],
            'confidence':   confidence,
            'geometric':    geometric_feats,
            'color_feats':  color_feats,
            'texture_feats':texture_feats,
            'result_image': '/' + result_path.replace('\\','/')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)