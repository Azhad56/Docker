from flask import Flask,request
import numpy as np
import pickle
import pandas as pd
app=Flask(__name__)


pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


@app.route("/")
def welcome():
    return "welcome to docker tutorials"

@app.route('/predict/',methods=["Get"])
def predict_note_authentication():

    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "Hello The answer is"+str(prediction)

@app.route('/predict_file/',methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    print(df_test.head())
    return str(list(prediction))
if __name__ == "__main__":
    app.run(port = 5000)