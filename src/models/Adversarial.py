import datetime
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, vstack
import os

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# read train and test data from csv
train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')

train['TARGET'] = 0
test['TARGET'] = 1

# set directory for pickling
directory = 'data/processed/used/'

# initialize dictionary for reading presaved vectors
train_vectors = {}
test_vectors = {}

# read vectors from directory and save in dictionaries
for file in os.listdir(directory):
    if file.endswith(".sav"):
        name = file.split(".")[0]
        if name.startswith("train"):
            train_vectors[name] = pickle.load(open(directory+file, 'rb'))
        elif name.startswith("test"):
            test_vectors[name] = pickle.load(open(directory + file, 'rb'))

# hstack the vectors
train_features = hstack([train_vectors[vec] for vec in train_vectors])
test_features = hstack([test_vectors[vec] for vec in test_vectors])

# initialize scores and submission variables
scores = []
classifier = {}
submission = pd.DataFrame.from_dict({'id': test['id']})

X = vstack([train_features,test_features])
y = pd.concat([train['TARGET'],test['TARGET']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = LogisticRegression(solver='sag')

classifier.fit(X, y)
adversarial = pd.DataFrame(y.values,columns=['y'])
adversarial['predict'] = classifier.predict_proba(X)[:, 1]

ranges = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
a = adversarial[adversarial.y==0].sort_values(by='predict')
b = a.groupby(pd.cut(a.predict, ranges)).count()
