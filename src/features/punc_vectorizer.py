import string
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])


def get_punc(message):
    only_punc = [i for i in message.split() if all(j in string.punctuation for j in i)]
    return only_punc


punc_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer=get_punc)
punc_vectorizer.fit(all_text)
print('Finished fitting punc_vectorizer')
train_punc_features = punc_vectorizer.transform(train_text)
test_punc_features = punc_vectorizer.transform(test_text)
print('Finished transforming punc_vectorizer')

pickle.dump(train_punc_features, open('src/data/train_punc_features.sav', 'wb'))
pickle.dump(test_punc_features, open('src/data/test_punc_features.sav', 'wb'))
