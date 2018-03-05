
import datetime
import pandas as pd
import os

directory = 'src/models/'
submissions = {}
columns = {}
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# set the time to uniquely identify the model
str_time = datetime.datetime.now().strftime('%Y%m%d%I%M')

# read vectors from directory and save in dictionaries
for file in os.listdir(directory):
    if file.endswith(".csv"):
        submissions[file] = pd.read_csv(directory+file)

real_sub = next(iter(submissions.values()))

for col in class_names:
    columns[col] = pd.concat([i[col] for i in submissions.values()],axis=1)
    columns[col] = columns[col].mean(axis=1)
    real_sub[col] = columns[col]

real_sub.to_csv('src/models/submission'+str_time+'.csv', index=False)


