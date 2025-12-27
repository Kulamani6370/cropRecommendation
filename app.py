from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standardscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = [
        data['Nitrogen'],
        data['Phosphorus'],
        data['Potassium'],
        data['Temperature'],
        data['Humidity'],
        data['Ph'],
        data['Rainfall']
    ]

    single_pred = np.array(features).reshape(1, -1)
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    prediction = model.predict(final_features)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton",
        5: "Coconut", 6: "Papaya", 7: "Orange", 8: "Apple",
        9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
        12: "Mango", 13: "Banana", 14: "Pomegranate",
        15: "Lentil", 16: "Blackgram", 17: "Mungbean",
        18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    crop = crop_dict.get(prediction[0], "Unknown")

    return jsonify({
        "recommended_crop": crop
    })

if __name__ == "__main__":
    app.run()
