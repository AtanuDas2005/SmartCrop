from flask import Flask,request,jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# load trained model
model = pickle.load(open("crop_model.pkl","rb"))

soil_map = {
"loamy":0,
"sandy":1,
"clay":2,
"black":3
}

season_map = {
"summer":0,
"winter":1,
"monsoon":2
}

@app.route("/predict",methods=["POST"])
def predict():

    data = request.json

    soil = soil_map[data["soil"]]
    temp = float(data["temperature"])
    rain = float(data["rainfall"])
    season = season_map[data["season"]]

    prediction = model.predict([[soil,temp,rain,season]])

    return jsonify({"crop":prediction[0]})


if __name__ == "__main__":
    app.run(debug=True)