
import os
import datetime as dt
import matplotlib.pyplot as plt

import json

from keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed, Masking

import tensorflow as tf
from keras import optimizers, Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed


class LSTMAutoEncoder():
    def __init__(self, name, outcome, timesteps, n_features):
        self.lstm_autoencoder = Sequential(name = name)
        # Encoder
        self.lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
        self.lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
        self.lstm_autoencoder.add(Dropout(0.5))

        self.lstm_autoencoder.add(RepeatVector(timesteps))
        # Decoder
        self.lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
        #self.lstm_autoencoder.add(Dropout(0.5))
        self.lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
        self.lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
        lr = 0.001
        adam = optimizers.Adam(lr)
        (self.lstm_autoencoder).compile(loss='mse', optimizer=adam)

        self.outcome = outcome
        self.history = None

    def summary( self ):
        self.lstm_autoencoder.summary()



    def fit(self, trainx, trainy,e, b,val_x, val_y,v):
        configs = json.load(open('Configuration.json', 'r'))

        configs["paths"]["autoencoder_path"]
        name =  os.path.join('%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(e)))
        history = self.lstm_autoencoder.fit(trainx, trainy,epochs = e, batch_size = b,
                          validation_data = (val_x,val_y), verbose = v).history
        self.lstm_autoencoder.save(name)
        self.history = history
      #self.history = history

    def plot_history( self ):
        plt.figure(figsize=(10, 10))
        plt.plot(self.history['loss'], linewidth=2, label='Train')
        plt.plot(self.history['val_loss'], linewidth=2, label='Valid')
        plt.legend(loc='upper right')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        configs = json.load(open('Configuration.json', 'r'))
        autoencoder_path = configs['paths']['autoencoder_path']

        plt.savefig(autoencoder_path+self.outcome+self.outcome+"LossOverEpochs.pdf", bbox_inches='tight')

        plt.figure(figsize=(10, 10))

    def predict( self , xval):
        predictions = self.lstm_autoencoder.predict(xval)
        return predictions






