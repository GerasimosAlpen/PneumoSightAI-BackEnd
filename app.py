from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import logging

# TensorFlow & Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# ========================
# Config
# ========================
MODEL_PATH = "chestXrayModelRevision.keras"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================
# Logging setup
# ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# Lazy load model
# ========================
_model = None
def get_model():
    global _model
    if _model is None:
        logger.info("Loading AI model...")
        _model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
    return _model

# ========================
# Predict function
# ========================
def predict_image(image_path: str) -> dict:
    model = get_model()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.resize(img, (150, 150))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=-1)  # pastikan shape (1,150,150,1)

    pred = float(model.predict(img, verbose=0)[0][0])

    if pred >= 0.5:
        return {"prediction": "Normal", "probability": round(pred, 2)}
    else:
        return {"prediction": "Pneumonia", "probability": round(1 - pred, 2)}

# ========================
# Flask App
# ========================
def create_app():
    app = Flask(__name__)

    # Setup CORS
    CORS(app, resources={
        r"/*": {
            "origins": [
                "http://localhost:5173",
                "https://pneumo-sight-rlf4ff5ed-alvens-projects-cf34feb6.vercel.app"
            ]
        }
    })

    # Health check
    @app.route("/", methods=["GET"])
    def health():
        return "OK", 200

    # Predict endpoint
    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        try:
            result = predict_image(path)
            return jsonify(result)
        except Exception as e:
            logger.exception("Prediction error")
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(path):
                os.remove(path)

    return app

# ========================
# Run App
# ========================
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port)
