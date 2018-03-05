import datetime
import numpy as np
import pandas as pd
import pickle
import os

from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, GRU
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
maxlen = 100

# set directory for pickling
directory = 'src/data/'


# read train and test data from csv
train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')


list_sentences_train = train["comment_text"].fillna("CVxTz").values
class_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[class_names].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()
batch_size = 32
epochs = 2
file_path = directory+"weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

model.load_weights(file_path)

y_test = model.predict(X_te)

submission = pd.read_csv("data/raw/sample_submission.csv.zip")

submission[class_names] = y_test

# set the time to uniquely identify the model
str_time = datetime.datetime.now().strftime('%Y%m%d%I%M')
# save the model details
model_details = {
    'name': str_time,
    'classifier': model,
    # 'scores': pd.DataFrame(scores,index=class_names,columns=['roc_score']),
    #'vectors': [vec for vec in train_vectors]
}

submission.to_csv('src/models/submission'+str_time+'.csv', index=False)


