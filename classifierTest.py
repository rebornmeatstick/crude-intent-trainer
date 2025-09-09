from scipy.special import modfresnelp
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
import numpy as np
import re
import os
import joblib

# path
mfp = os.path.abspath(__file__)
mfd = os.path.dirname(mfp)

# load model
classifierModel = input("enter the model name please! :) || [input] : ")

# add .pkl if not present
match_pkl = re.search(r"^[^.]+\.pkl$", classifierModel)
if match_pkl == None:
    classifierModel = classifierModel + ".pkl"
modelPath = f"{mfd}/models/{classifierModel}"

clf = joblib.load(modelPath)

def predictionFunc():
    prediction_text = input("text to predict with? || [input] : ")
    prediction = clf.predict([prediction_text])
    confidence = clf.predict_proba([prediction_text])
    print(f"-0prediction: {prediction}")
    print(f"confidence: {np.max(confidence):.2f}")

    predictionFunc()

predictionFunc()
