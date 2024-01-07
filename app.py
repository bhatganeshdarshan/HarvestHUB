from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import sklearn

prediction_model = pickle.load(open("prediction_model.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/crop_prediction',methods=['POST'])
def prediction():
    n=request.form['Nitrogen']
    p=request.form['Phosphorous']
    k=request.form['Potassium']
    temp=request.form['Temperature']
    hum=request.form['Humidity']
    pH = request.form['pH']
    rf = request.form['Rainfall']
    featured_data = {'columns':list('N','P','K','temperature','humidity','ph','rainfall')}
    test_series = pd.Series(np.zeros(7),index=featured_data['columns'])
    test_series['N']=n
    test_series['P']=p
    test_series['K']=k
    test_series['temperature'] =temp
    test_series['humidity']=hum
    test_series['ph']=pH
    test_series['rainfall'] =rf
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    predict_crop = prediction_model.predict(test_series)
    if predict_crop in crop_dict:
        crop = crop_dict[predict_crop]
        res = f"{crop} is the best crop to cultivate"
    else:
        res = "No suitable crop found for the given data !"
    return render_template('index.html',result=res)

if __name__=="__main__":
    app.run(debug=True)

