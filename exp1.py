from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(12)
tf.random.set_seed(12)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed, Reshape
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import DataStore

ds = DataStore.DataStore("data", 100)
X_train = ds.trainData['A-3'].x
Y_train = ds.trainData['A-3'].y
Y_train = Y_train.reshape(X_train.shape[0],1,25)

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

print("starting training")
history = model.fit(X_train, Y_train, epochs=500, batch_size=1, validation_split=0.2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, mode='min')], shuffle=False)

model.save("models/experimental-1/A-3.h5")
print("experiment complete\n*10")
