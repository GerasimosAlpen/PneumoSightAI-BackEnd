from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import cv2
import numpy as np

def create_app():
    app = Flask(__name__)
    CORS(app, origins=[
        "http://localhost:5173",
        "https://pneumo-sight-rlf4ff5ed-alvens-projects-cf34feb6.vercel.app"
    ])

    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    @app.route("/", methods=["GET"])
    def health():
        return "OK", 200   # ⬅️ HARUS STRING, JANGAN JSON

    @app.route("/predict", methods=["POST"])
    def predict():
        from model import predict_image   # ⬅️ IMPORT DI SINI

        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400

        file = request.files["file"]
        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        try:
            label, prob = predict_image(path)
        finally:
            os.remove(path)

        return jsonify({
            "prediction": label,
            "probability": round(prob, 2)
        })

    return app


app = create_app()
