from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app,origins=["http://localhost:5173"])

# Cargar el modelo

# Clases en el mismo orden que en el entrenamiento
CLASSES = ["melanoma", "normal_skin", "psoriasis"]

@app.route('/')
def index():
    return "Hola desde Flask"

@app.route("/predict", methods=["POST"])
def predict():
    model = load_model("model/modelo_skin.h5")
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    image = Image.open(io.BytesIO(image.read())).convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)[0]
    class_index = np.argmax(predictions)
    class_label = CLASSES[class_index]
    confidence = float(predictions[class_index])

    return jsonify({
        "class": class_label,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
