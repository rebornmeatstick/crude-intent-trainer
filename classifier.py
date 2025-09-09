from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
import re
import json
import os
import joblib
import numpy as np

# variables
texts = []
labels = []
classes = []

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

# import .json file with training data

filename = input("enter .json filename containing training data please! :) || [input] : ")

# add .json if not present
match_json = re.search(r"^[^.]+\.json$", filename)
if match_json == None:
    filename = filename + ".json"

filepath = f"{mfd}/data/{filename}"

with open(filepath, "r") as file:
    global training_data
    try:
        training_data = json.load(file)
    except:
        print("couldn't open .json file ... did you type in the wrong name? exiting now ...")
        exit(67)

# seperate texts, labels, and classes into different sections
classes = training_data["classes"]
for item in training_data["data"]:
    texts.append(item["texts"])
    labels.append(item["labels"])
    print(f'seperated a {item["labels"]} label')

# training stuff

if not os.path.isfile(modelPath):
    print("-- entered model name doesn't exist! will create after training! --")
    clf = make_pipeline(
    TfidfVectorizer(),
    SGDClassifier(loss="log_loss"),
    )
else:
    clf = joblib.load(modelPath)



clf.fit(texts, labels)

#clf.named_steps['sgdclassifier'].partial_fit(
#    X,
#    labels,
#    classes=classes,
#    )

# save model
joblib.dump(clf, modelPath)
print("model dumped")