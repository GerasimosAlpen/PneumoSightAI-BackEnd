from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

def create_app():
    app = Flask(__name__)

    # CORS FINAL (AMAN UNTUK DEV + VERCEL)
    CORS(
        app,
        origins=[
            "http://localhost:5173",
            "https://pneumo-sight-rlf4ff5ed-alvens-projects-cf34feb6.vercel.app"
        ],
        methods=["GET", "POST"],
        allow_headers=["Content-Type"]
    )

    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    @app.route("/", methods=["GET"])
    def health():
        return "OK", 200  # ⚠️ HARUS STRING

    @app.route("/predict", methods=["POST"])
    def predict():
        from model import predict_image  # IMPORT DI SINI (SAFE FOR GUNICORN)

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        try:
            result = predict_image(path)
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(path):
                os.remove(path)

    return app


app = create_app()
