"""
Flask app for the chest X-ray COVID classifier.

Run locally:      python flask_app.py
Run in production: gunicorn -w 1 --threads 4 -b 0.0.0.0:$PORT flask_app:app

Configuration is by environment variable, so nothing is hard-coded to one PC:
  MODEL_PATH         path to the model file (.keras, .h5 or .tflite)
  MODEL_URL          optional; downloaded to MODEL_PATH on first start if missing
  COVID_CLASS_INDEX  which sigmoid output means COVID: 0 or 1 (see README note)
"""
import io
import os
import threading
import urllib.request

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(BASE_DIR, "models", "Covid_hybrid_v2.keras")
)
MODEL_URL = os.environ.get("MODEL_URL")

IMG_SIZE = 224          # must match IMG_SIZE in the training script
THRESHOLD = 0.5
MAX_UPLOAD_MB = 10
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "WEBP"}

# flow_from_directory numbers classes alphabetically, and the model's single
# sigmoid output is the probability of class index 1.  Your original app treated
# "output < 0.5" as COVID, i.e. COVID = index 0, so that is the default here.
# Confirm with:  print(train_gen.class_indices)  in the training script.
COVID_CLASS_INDEX = int(os.environ.get("COVID_CLASS_INDEX", "0"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
class Predictor:
    """Wraps a Keras model or a TFLite model behind one predict() call."""

    def __init__(self, path):
        self._lock = threading.Lock()  # TFLite interpreters are not thread-safe
        self._tflite = path.endswith(".tflite")
        if self._tflite:
            try:  # slim runtime: pip install ai-edge-litert (TensorFlow not needed)
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf

                Interpreter = tf.lite.Interpreter
            self._interp = Interpreter(model_path=path)
            self._interp.allocate_tensors()
            self._in = self._interp.get_input_details()[0]["index"]
            self._out = self._interp.get_output_details()[0]["index"]
        else:
            import tensorflow as tf

            self._model = tf.keras.models.load_model(path, compile=False)

    def predict(self, batch):
        """Returns the raw sigmoid output: P(class index 1)."""
        with self._lock:
            if self._tflite:
                self._interp.set_tensor(self._in, batch)
                self._interp.invoke()
                return float(self._interp.get_tensor(self._out)[0][0])
            return float(self._model.predict(batch, verbose=0)[0][0])


def load_predictor():
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Put it there, or set MODEL_URL."
            )
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Downloading model from {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return Predictor(MODEL_PATH)


predictor = None
load_error = None
try:
    predictor = load_predictor()
    print("Model loaded:", MODEL_PATH)
except Exception as exc:  # keep the server up so /health can say what is wrong
    load_error = str(exc)
    print("Model failed to load:", load_error)


# --------------------------------------------------------------------------- #
# Image handling
# --------------------------------------------------------------------------- #
def read_image(file_storage):
    """Validate and decode an upload. Nothing is written to disk."""
    try:
        img = Image.open(io.BytesIO(file_storage.read()))
        fmt = img.format
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("That file could not be read as an image.")
    if fmt not in ALLOWED_FORMATS:
        raise ValueError("Unsupported image type. Use PNG, JPG, BMP or WebP.")
    return img


def preprocess(img):
    # NEAREST matches Keras' default for both load_img and flow_from_directory,
    # so inference sees the same pixels the model saw in training.
    rgb = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.asarray(rgb, dtype="float32")
    return np.expand_dims(arr / 255.0, axis=0), arr


def looks_colourful(arr, tolerance=12.0):
    """X-rays are greyscale. Strong colour usually means the wrong kind of photo."""
    spread = np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2])
    return float(spread.mean()) > tolerance


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    ok = predictor is not None
    return jsonify({"ok": ok, "error": load_error}), (200 if ok else 503)


@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "The model is not available on the server."}), 503

    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file uploaded."}), 400
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        img = read_image(file)
        batch, raw_pixels = preprocess(img)
        raw = predictor.predict(batch)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The model could not process this image."}), 500

    p_covid = raw if COVID_CLASS_INDEX == 1 else 1.0 - raw
    is_covid = p_covid >= THRESHOLD

    warnings = []
    if looks_colourful(raw_pixels):
        warnings.append(
            "This image has a lot of colour. Chest X-rays are greyscale, so the "
            "result may not mean anything."
        )

    return jsonify(
        {
            "label": "covid" if is_covid else "non-covid",
            "result": "COVID" if is_covid else "Non-COVID",  # kept for the old frontend
            "covid_probability": round(p_covid, 4),
            "threshold": THRESHOLD,
            "warnings": warnings,
        }
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"The image is larger than {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",  # never leave debug on when hosted
    )
