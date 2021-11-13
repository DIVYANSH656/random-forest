from flask import Flask,render_template,request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import numpy as np
from pickle import load
app = Flask(__name__)
@app.route('/',methods=["GET"])
@cross_origin()
def home_page():
    return render_template("index.html")

@app.route('/predict',methods=["POST","GET"])
@cross_origin()
def index():
    if request.method == "POST":
        try:
            crim = float(request.form['crim'])
            zn = float(request.form['zn'])
            indus = float(request.form['indus'])
            chas = float(request.form['chas'])
            nox = float(request.form['nox'])
            rm = float(request.form['rm'])
            age = float(request.form['age'])
            dis = float(request.form['dis'])
            rad = float(request.form['rad'])
            tax = float(request.form['tax'])
            ptratio = float(request.form['ptratio'])
            b = float(request.form['b'])
            lstat = float(request.form['lstat'])
            filename = 'rfmdoel.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            st_sc = pickle.load(open("standardscaler.sav", 'rb'))
            prediction = loaded_model.predict(st_sc.transform([[crim, zn, indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]]))
            print("prediction is ",prediction)
            return render_template('result.html', prediction=round(prediction[0],3))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)