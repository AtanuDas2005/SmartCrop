
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# =====================================
# FLASK CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=".",
    static_folder="static"
)

CORS(app)


# =====================================
# LOAD CROP RECOMMENDATION MODEL
# =====================================

with open(BASE_DIR / "crop_model.pkl", "rb") as file:
    model = pickle.load(file)


soil_map = {
    "loamy": 0,
    "sandy": 1,
    "clay": 2,
    "black": 3
}

season_map = {
    "summer": 0,
    "winter": 1,
    "monsoon": 2
}


# =====================================
# LOAD TOMATO CNN MODEL
# =====================================

cnn_model = tf.keras.models.load_model(
    BASE_DIR / "models" / "tomato_model.keras"
)

with open(
    BASE_DIR / "models" / "class_names.json",
    "r",
    encoding="utf-8"
) as file:

    disease_classes = json.load(file)


# =====================================
# DASHBOARD ROUTE
# =====================================

@app.route("/")
@app.route("/dashboard.html")
def dashboard():

    return render_template("dashboard.html")


# =====================================
# PEST DETECTION PAGE
# =====================================

@app.route("/pest")
@app.route("/pest.html")
def pest_page():

    return render_template("pest.html")


# =====================================
# OTHER SMARTCROP PAGES
# =====================================

@app.route("/crop.html")
def crop_page():

    return render_template("crop.html")


@app.route("/Weather.html")
def weather_page():

    return render_template("Weather.html")


@app.route("/equipment.html")
def equipment_page():

    return render_template("equipment.html")


@app.route("/MarketPrices.html")
def market_page():

    return render_template("MarketPrices.html")


# =====================================
# CROP RECOMMENDATION API
# =====================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data received."
            }), 400

        soil = soil_map[data["soil"].lower()]

        temp = float(data["temperature"])

        rain = float(data["rainfall"])

        season = season_map[data["season"].lower()]

        prediction = model.predict([
            [soil, temp, rain, season]
        ])

        return jsonify({
            "crop": str(prediction[0])
        })

    except KeyError as e:

        return jsonify({
            "error": f"Invalid or missing input: {str(e)}"
        }), 400

    except Exception:

        app.logger.exception("Crop prediction failed")

        return jsonify({
            "error": "Crop prediction failed."
        }), 500


# =====================================
# TOMATO DISEASE PREDICTION API
# =====================================

@app.route("/predict-disease", methods=["POST"])
def predict_disease():

    if "image" not in request.files:

        return jsonify({
            "error": "Please upload an image."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:

        return jsonify({
            "error": "No image selected."
        }), 400

    try:

        # Read uploaded image
        image = Image.open(
            uploaded_file.stream
        ).convert("RGB")

        # Resize image for CNN
        image = image.resize((224, 224))

        # Convert image to NumPy array
        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # CNN prediction
        predictions = cnn_model.predict(
            image_array,
            verbose=0
        )[0]

        # Highest prediction
        predicted_index = int(
            np.argmax(predictions)
        )

        # Top 3 predictions
        top_indices = np.argsort(
            predictions
        )[-3:][::-1]

        top_predictions = [

            {
                "class_name": disease_classes[int(index)],

                "score": round(
                    float(predictions[index]) * 100,
                    2
                )
            }

            for index in top_indices

        ]

        return jsonify({

            "predicted_class": disease_classes[
                predicted_index
            ],

            "score": round(
                float(predictions[predicted_index]) * 100,
                2
            ),

            "top_predictions": top_predictions

        })

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    except Exception:

        app.logger.exception(
            "CNN prediction failed"
        )

        return jsonify({
            "error": "Prediction failed. Check Flask terminal."
        }), 500


# =====================================
# RUN FLASK SERVER
# =====================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
