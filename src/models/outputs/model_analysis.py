import datetime
import pandas as pd
import pickle
from scipy.sparse import hstack
import os

output_directory = 'src/models/outputs/'
input_directory = 'data/processed/used/'
models = {}
# read train and test data from csv
train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')

for file in os.listdir(output_directory):
    if file.endswith(".sav"):
        a = pickle.load(open(output_directory+file, 'rb'))
        name = a['name']
        models[name] = a

test_vectors = {}

for file in os.listdir(input_directory):
    if file.endswith(".sav"):
        name = file.split(".")[0]
        if name.startswith("test"):
            test_vectors[name] = pickle.load(open(input_directory + file, 'rb'))

test_features = hstack([test_vectors[vec] for vec in test_vectors])

scores = models['201802230900']['scores'].copy()
for i in models:
    scores[i] = models[i]['scores']

scores.drop(columns=['roc_score'], inplace=True)
scoresT = scores.T

models['201802230433']['classifier'] = models['201802270846']['classifier']
models['201802230433']['vectors'] = models['201802270846']['vectors']

submission = pd.DataFrame.from_dict({'id': test['id']})
top_models = {}
for i in scoresT.columns:
    top_models[i] = {}
    a = scoresT[i].sort_values(ascending=False).head(1)
    top_models[i]['model'] = a.index.values[0]
    top_models[i]['score'] = a.values[0]
    top_models[i]['classifier'] = models[a.index.values[0]]['classifier'][i]
    top_models[i]['vectors'] = models[a.index.values[0]]['vectors']

    submission[i] = top_models[i]['classifier'].predict_proba(test_features)[:, 1]

# set the time to uniquely identify the model
str_time = datetime.datetime.now().strftime('%Y%m%d%I%M')
# dump the model details and submission file to disk
submission.to_csv('src/models/submission'+str_time+'.csv', index=False)
