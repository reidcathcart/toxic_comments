import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

train_text_flagged = train[train.iloc[:,2:].sum(axis=1)>0]['comment_text']


flagged_word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=20000)
flagged_word_vectorizer.fit(train_text_flagged)
print('Finished fitting flagged_word_vectorizer')
train_flagged_features = flagged_word_vectorizer.transform(train_text)
test_flagged_features= flagged_word_vectorizer.transform(test_text)
print('Finished transforming flagged_word_vectorizer')

pickle.dump(train_flagged_features, open('src/data/train_flagged_features.sav', 'wb'))
pickle.dump(test_flagged_features, open('src/data/test_flagged_features.sav', 'wb'))