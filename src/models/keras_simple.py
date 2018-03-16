import datetime
import numpy as np
import os
import pandas as pd
from scipy.sparse import hstack
import pickle
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
np.random.seed(89)

# region fit generators
def nn_batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter=0


def pred_batch_generator(X_data,batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(X_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        counter += 1
        yield np.array(X_batch)
        if (counter > number_of_batches):
            counter=0
# endregion

# region Read data and set variables
# read train and test data from csv
train = pd.read_csv('data/raw/train.csv.zip').fillna(' ')
test = pd.read_csv('data/raw/test.csv.zip').fillna(' ')
submission = pd.read_csv('data/raw/sample_submission.csv.zip')
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

# set the time to uniquely identify the model
str_time = datetime.datetime.now().strftime('%Y%m%d%I%M')

# set variables for files and directories
directory = 'data/processed/used/'
checkpoint_file = 'src/models/outputs/checkpoint'+str_time+'.h5'
sub_file = 'src/models/outputs/submission'+str_time+'.csv'

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
train_features = hstack([train_vectors[vec] for vec in train_vectors], format='csr')
test_features = hstack([test_vectors[vec] for vec in test_vectors], format='csr')

# shapes of features
train_cols = train_features.shape[1]
train_rows = train_features.shape[0]
test_cols = test_features.shape[1]
test_rows = test_features.shape[0]

# batch sizes
train_batch_size = 1024
val_batch_size = 1024
# endregion

# region Make the model
early_stop = EarlyStopping(patience=2)
checkpoint = ModelCheckpoint('src/models/outputs/checkpoint'+str_time+'.h5',
                             monitor='val_acc', verbose=1, save_best_only=True)

model = Sequential()
model.add(Dense(200, activation='sigmoid', input_shape=(train_cols,)))
model.add(Dense(200, activation='hard_sigmoid'))
model.add(Dense(6, activation="sigmoid"))
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
# endregion


# region Fit and predict
model.fit_generator(generator=nn_batch_generator(train_features, y_train, train_batch_size),
                    epochs=20, steps_per_epoch=train_rows/train_batch_size,
                    callbacks=[early_stop,checkpoint],
                    validation_data=nn_batch_generator(train_features, y_train, val_batch_size),
                    validation_steps=5)

model.load_weights(filepath=checkpoint_file)


y_pred = model.predict_generator(generator=pred_batch_generator(test_features,train_batch_size),
                                 steps=test_rows/train_batch_size)
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred


# save the model details
model_details = {
    'name': str_time,
    'classifier': model
}

submission.to_csv(sub_file, index=False)
# endregion


