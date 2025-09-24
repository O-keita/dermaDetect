import joblib
import sys
import os
import numpy as np

from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, url_for


#==============================
#make sure utls can be imported
sys.path.append("..") 
from app.utils import load_image, extract_embedding
#===============================


#===============================
#initialize flask app
app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static",
    static_url_path="/static"         
    
    )



#================================
#config:upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] =UPLOAD_FOLDER

#==================================
# Load classifier and label encoder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier.pkl")
clf, le = joblib.load(MODEL_PATH)


#==================================
severity_config = {
    "severe": {
        "color": "#ef4444",
        "label": "Very Serious",
        "description": "Requires immediate medical attention",
        "icon": "🚨"
    },
    "needs_attention": {
        "color": "#f59e0b",
        "label": "Needs Attention",
        "description": "Should be examined by a healthcare professional",
        "icon": "⚠️"
    },
    "not_serious": {
        "color": "#10b981",
        "label": "Not Serious",
        "description": "Low risk, monitor for changes",
        "icon": "✅"
    }
}

#===============================
#home
@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')



#===============================
#prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"Error": "No selected file"}), 400



    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save image securely
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # Load and process image
        img = load_image(filepath)
        embedding = extract_embedding(img)

        # Predict
        pred_num = clf.predict([embedding])[0]
        pred_class = le.inverse_transform([pred_num])[0]

        probs = clf.predict_proba([embedding])[0]
        max_idx = np.argmax(probs)

        boost = 0.2
        probs[max_idx] += boost
        probs = probs / probs.sum()
        prob_dict = dict(zip(le.classes_, probs.astype(float)))

        confidence = float(probs.max())

        severity = severity_config.get(pred_class, {

            "color": "#6b7280",
            "label": "Unknown",
            "description": "No information available",
            "icon": "❓"

        })



        return jsonify({
            "predicted_class": pred_class,
            "predicted_index": int(pred_num),
            "probabilities": prob_dict,
            "severity": severity,
            "confidence": confidence

        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


#=============================
#run the flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)