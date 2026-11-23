"""Iris Studio — Flask prediction app.

Run `python train_model.py` once if you do not already have iris.pkl.
"""
from pathlib import Path
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iris.pkl"
app = Flask(__name__)

FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")
SPECIES = {
    "0": "Iris setosa", "1": "Iris versicolor", "2": "Iris virginica",
    "setosa": "Iris setosa", "versicolor": "Iris versicolor", "virginica": "Iris virginica",
    "iris-setosa": "Iris setosa", "iris-versicolor": "Iris versicolor", "iris-virginica": "Iris virginica",
}


def species_name(value):
    """Convert numeric and textual class labels into readable species names."""
    if isinstance(value, np.generic):
        value = value.item()
    raw = str(value).strip()
    return SPECIES.get(raw.lower().replace("_", "-"), raw)


def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file iris.pkl not found. Run `python train_model.py` first, or copy your trained iris.pkl into this folder.")
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)  # Only load pickle files from trusted sources.


def parse_features(data):
    values = []
    for feature in FEATURES:
        if feature not in data:
            raise ValueError(f"Missing measurement: {feature.replace('_', ' ')}")
        try:
            value = float(data[feature])
        except (TypeError, ValueError):
            raise ValueError(f"Enter a valid number for {feature.replace('_', ' ')}") from None
        if not np.isfinite(value) or not 0 < value <= 15:
            raise ValueError(f"{feature.replace('_', ' ').capitalize()} must be greater than 0 and at most 15 cm")
        values.append(value)
    return values


def prediction_result(values):
    model = get_model()
    row = np.asarray([values], dtype=float)
    label = species_name(model.predict(row)[0])
    result = {"species": label, "measurements": dict(zip(FEATURES, values)), "probabilities": None}
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(row)[0], dtype=float)
        classes = getattr(model, "classes_", np.arange(len(probs)))
        if len(classes) == len(probs) and np.isfinite(probs).all():
            result["probabilities"] = [
                {"species": species_name(cls), "probability": round(float(prob) * 100, 1)}
                for cls, prob in sorted(zip(classes, probs), key=lambda item: float(item[1]), reverse=True)
            ]
    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("info.html", heading="About Iris Studio", eyebrow="THE PROJECT", body="Iris Studio uses a trained machine-learning model to classify iris flowers into three species using four measurements: sepal length, sepal width, petal length and petal width. Prediction probabilities are displayed only when the model supports them.")


@app.route("/profile")
def profile():
    return render_template("info.html", heading="Project profile", eyebrow="MODEL & METHODOLOGY", body="This application runs a local scikit-learn-compatible model saved as iris.pkl. It accepts measurements in centimetres and makes predictions on the server. Replace iris.pkl with your own trusted trained model to use your own classifier.")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        values = parse_features(request.get_json(silent=True) or {})
        return jsonify(prediction_result(values))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction is unavailable. Ensure iris.pkl is installed and compatible with this Python environment."}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """Preserves compatibility with the user's original HTML form endpoint."""
    try:
        data = request.get_json(silent=True) if request.is_json else request.form
        values = parse_features(data or {})
        result = prediction_result(values)
        if request.is_json:
            return jsonify(result)
        return render_template("index.html", initial_result=result, initial_values=result["measurements"])
    except ValueError as error:
        if request.is_json:
            return jsonify({"error": str(error)}), 400
        return render_template("index.html", form_error=str(error)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        if request.is_json:
            return jsonify({"error": "Prediction is temporarily unavailable."}), 503
        return render_template("index.html", form_error="Unable to load the trained model. Check iris.pkl."), 503


if __name__ == "__main__":
    app.run(debug=True)
