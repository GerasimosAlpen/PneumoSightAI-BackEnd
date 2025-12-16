from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

# PASTIKAN model.py ada di folder yang sama
from model import predict_image, get_model

# Load model di startup biar ga delay pas request pertama
print("Loading AI model at startup...")
get_model()
print("Model loaded successfully!")

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

    # Folder upload sementara
    UPLOAD_FOLDER = "uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Health check endpoint
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
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(path):
                os.remove(path)

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
