from flask import Flask,render_template, request
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)
data= pd.read_csv('clean_data.csv')
pipe=pickle.load(open("RidgeModel.pkl", "rb"))
@app.route('/')
def index():
    locations=sorted(data['location'].unique())
    return render_template("index.html", locations=locations, title="Rooflyzer")
@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = request.form.get('sqft')
    input=pd.DataFrame([[locations, sqft, bhk, bath]], columns=['location', 'total_sqft', 'bhk', 'bath'])
    prediction=pipe.predict(input)[0] *1e5
    return str(np.round(prediction,2))
if __name__ == '__main__':
    app.run(debug=True, port=5000)