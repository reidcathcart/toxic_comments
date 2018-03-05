import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=20000)
char_vectorizer.fit(all_text)
print('Finished fitting char_vectorizer')
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
print('Finished transforming char_vectorizer')

pickle.dump(train_char_features, open('src/data/train_char_features.sav', 'wb'))
pickle.dump(test_char_features, open('src/data/test_char_features.sav', 'wb'))
