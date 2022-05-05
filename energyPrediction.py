from flask import Flask, render_template, request, json, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import requests
import numpy as np
import pandas as pd
import sklearn
import pickle


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def load_model():
    global model
    with open('energymodel.pkl', 'rb') as f:
        model = pickle.load(f)



def predict(data):
    #get data from request
    data = request.get_json(force=True)
    data_categoric = np.array([data["meter"], data["primary_use"], data["square_feet"], data["year_built"], data["air_temperature"], data["cloud_coverage"], data["dew_temperature"], data["hour"], data["day"], data["weekend"], data["month"]])
    data_categoric = np.reshape(data_categoric, (1, -1))
   
    data_age = np.array(StandardScaler().fit_transform(data_categoric))

    data_final = pd.DataFrame(data_final, dtype=object)

    #make predicon using model
    prediction = model.predict(data_final)
    return prediction[0]


app = Flask(__name__)

#decorator to access the app
@app.route("/")
@app.route("/index")
def index():
    return render_template("home.html")

#decorator to access the service
@app.route("/energyPrediction", methods=['GET', 'POST'])
def energyPrediction():

    if request.method == 'GET':
        return (flask.render_template('home.html'))
    
    if request.method == 'POST':
        #extract from inputs
        meter = request.form.get("meter")
        primary_use = request.form.get("primary_use")
        square_feet = request.form.get("square_feet")
        year_built = request.form.get("year_built")
        air_temperature = request.form.get("air_temperature")
        cloud_coverage = request.form.get("cloud_coverage")
        dew_temperature = request.form.get("dew_temperature")
        hour = request.form.get("hour")
        day = request.form.get("day")
        weekend = request.form.get("weekend")
        month = request.form.get("month")

        input_data = pd.DataFrame([[meter, primary_use, square_feet, year_built, air_temperature, cloud_coverage, dew_temperature, hour, day, weekend, month]],
        columns=['meter', 'primary_use', 'square_feet', 'year_built', 'air_temperature', 'cloud_coverage', 'dew_temperature', 'hour', 'day', 'weekend', 'month'], dtype = float)

        

   #convert data to json
    #data = json.dumps({"meter": meter, "primary_use": primary_use, "square_feet": square_feet, "year_built": year_built, "air_temperature": air_temperature, "cloud_coverage": cloud_coverage, "dew_temperature": dew_temperature, "hour": hour, "day": day, "weekend": weekend, "month": month})
    #input_d = np.array([meter, primary_use, square_feet, year_built, air_temperature, cloud_coverage,
    #dew_temperature, hour, day, weekend, month])
    #url for bank marketing model
    #url = "http://localhost:5000/api"
    #url = "http://0.0.0.0:8080/mlapi"

       #get data from request
    #data = request.get_json(force=True)
    #data_categoric = np.array([data["meter"], data["primary_use"], data["square_feet"], data["year_built"], data["air_temperature"], data["cloud_coverage"], data["dew_temperature"], data["hour"], data["day"], data["weekend"], data["month"]])
    #data_categoric = np.reshape(data_categoric, (1, -1))
   
        data_final = np.array(StandardScaler().fit_transform(input_data))

    

    #make predicon using model
    #prediction = model.predict(data_final)
        prediction = model.predict(input_data)[0]
  
    #post data to url
        results =  prediction

    #send input values and prediction result to index.html for display
        return render_template("home.html", meter = meter, primary_use = primary_use, square_feet = square_feet, year_built = year_built, air_temperature = air_temperature, cloud_coverage = cloud_coverage, dew_temperature = dew_temperature, hour = hour, day = day, weekend = weekend, month = month, results=prediction)
  
if __name__=="__main__":
    load_model()
    app.run(host='0.0.0.0', port=8000)
