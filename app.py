from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle

prediction_model = pickle.load(open("prediction_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/crop_prediction", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosphorous']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    prediction = prediction_model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to cultivate"
    else:
        result = "Could not find any suitable crops for this data"

    return redirect(url_for('result_page', result=result, crop=crop))

@app.route('/result_page')
def result_page():
    result = request.args.get('result')
    crop = request.args.get('crop')
    return render_template('result_page.html', result=result, crop=crop)

if __name__ == "__main__":
    app.run(debug=True)
