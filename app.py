from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

app = Flask(__name__)

CORS(
    app,
    origins=[
        "http://localhost:5173",
        "https://pneumo-sight-rlf4ff5ed-alvens-projects-cf34feb6.vercel.app"
    ],
    methods=["GET", "POST"]
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "chestXrayModelRevision.h5"
model = None


def get_model():
    global model
    if model is None:
        print("Loading AI model...")
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return model


def predict_image(image_path):
    import cv2
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array

    model = get_model()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image cannot be read")

    img = cv2.resize(img, (150, 150))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred >= 0.5:
        return "Normal", float(pred)
    else:
        return "Pneumonia", float(1 - pred)


@app.route("/", methods=["GET"])
def health_check():
    return {
        "status": "ok",
        "message": "Pneumosight AI Backend is running"
    }, 200


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        prediction, probability = predict_image(path)
    except Exception as e:
        os.remove(path)
        return jsonify({"error": str(e)}), 500

    os.remove(path)

    return jsonify({
        "prediction": prediction,
        "probability": round(probability, 2)
    })
