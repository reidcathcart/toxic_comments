import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=20000)
word_vectorizer.fit(all_text)
print('Finished fitting word_vectorizer')
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
print('Finished transforming word_vectorizer')

pickle.dump(train_word_features, open('src/data/train_word_features.sav', 'wb'))
pickle.dump(test_word_features, open('src/data/test_word_features.sav', 'wb'))

