import datetime
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import os

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# read train and test data from csv
train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')

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

# train model for each flag using LogisticRegression
# score using cross_val_score
for class_name in class_names:
    train_target = train[class_name]
    classifier[class_name] = LogisticRegressionCV(
        penalty='l2',
        scoring='roc_auc',
        cv=3,
        solver='sag',
        verbose=0
    )

    classifier[class_name].fit(train_features, train_target)
    cv_score = classifier[class_name].scores_[1].mean(axis=0).max()
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))
    submission[class_name] = classifier[class_name].predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

# set the time to uniquely identify the model
str_time = datetime.datetime.now().strftime('%Y%m%d%I%M')
# save the model details
model_details = {
    'name': str_time,
    'classifier': classifier,
    'scores': pd.DataFrame(scores,index=class_names,columns=['roc_score']),
    'vectors': [vec for vec in train_vectors]
}

# dump the model details and submission file to disk
pickle.dump(model_details, open('src/models/model_details'+str_time+'.sav', 'wb'))
submission.to_csv('src/models/submission'+str_time+'.csv', index=False)
