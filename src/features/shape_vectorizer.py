import pandas as pd
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer

train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

nlp = spacy.load('en_core_web_lg')
regexp_tokenizer = RegexpTokenizer(r'\w+')


def get_shape(message):
    only_words = regexp_tokenizer.tokenize(message)
    spacy_tokenizer = spacy.tokenizer.Tokenizer(nlp.vocab)
    filtered_message = spacy_tokenizer(' '.join(only_words))
    filtered_message_list = list([t.shape_ for t in filtered_message])
    return filtered_message_list

shape_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer=get_shape,
    stop_words=None,
    max_features=20000)
shape_vectorizer.fit(train_text_flagged)
print('Finished fitting shape_vectorizer')
train_shape_features = shape_vectorizer.transform(train_text)
test_shape_features = shape_vectorizer.transform(test_text)
print('Finished transforming shape_vectorizer')

pickle.dump(train_shape_features, open('src/data/train_shape_features.sav', 'wb'))
pickle.dump(test_shape_features, open('src/data/test_shape_features.sav', 'wb'))