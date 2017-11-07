"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 


To check the Keras version:
python -c "import keras; print keras.__version__"
"""

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse
import json
import time
from math import sqrt
from keras.callbacks import Callback

__author__ = 'bejar'

class MeanSquaredErrorHistory(Callback):
        def on_train_begin(self, logs={}):
                self.losses=[]
        def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('mse'))

class LossHistory(Callback):
        def on_train_begin(self, logs={}):
                self.losses = []
        def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))

def lagged_vector(data, lag=1):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    for i in range(lag):
        lvect.append(data[i: -lag+i])
    lvect.append(data[lag:])
    return np.stack(lvect, axis=1)

def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gru implementation", action='store_true', default=False)
    args = parser.parse_args()
	
    ModelName='RNN'

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 0

    config = load_config_file(args.config)

    print("Starting:", time.ctime())

    ############################################
    # Data

    vars = {0: 'wind_speed', 1: 'air_density', 2: 'temperature', 3: 'pressure'}

    wind = np.load('Wind.npz')
    print(wind.files)
    # ['90-45142', '90-45143', '90-45230', '90-45229']
    m1 = wind['90-45142']
    m1 = m1[:, 0:3]

    m2 = wind['90-45143']
    m2 = m2[:, 0:3]

    m3 = wind['90-45230']
    m3 = m3[:, 0:3]

    m4 = wind['90-45229']
    m4 = m4[:, 0:3]

    data = np.concatenate((m1, m2, m3, m4))
    scaler = StandardScaler()
    wind = scaler.fit_transform(data)

    # Size of the training and size for validation + test set (half for validation, half for test)
    datasize = config['datasize']
    testsize = config['testsize']

    # Length of the lag for the training window
    lag = config['lag']

    wind_train = wind[:datasize, 0]
    train = lagged_vector(wind_train, lag=lag)
    train_x, train_y = train[:, :-1], train[:,-1]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    wind_test = wind[datasize:datasize+testsize, 0]
    test = lagged_vector(wind_test, lag=lag)
    half_test = int(test.shape[0]/2)

    val_x, val_y = test[:half_test, :-1], test[:half_test,-1]
    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))

    test_x, test_y = test[half_test:, :-1], test[half_test:,-1]
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    #Calculate the baseline root mean squared error of the persistence model for the validation set(i.e. "walk-forward validation")
    #Compare: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
    historyX = [y for y in train_y]
    predictions = list()
    for i in range(len(val_y)):
        # make prediction
        predictions.append(historyX[-1])
        # observation
        historyX.append(val_y[i])
    #report performance
    rmse = sqrt(mean_squared_error(val_y, predictions))
    print('Benchmark RMSE validation set: %.3f' % rmse)

    #Calculate the baseline root mean squared error of the persistence model for the test set(i.e. "walk-forward validation")
    #Compare: https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
    historyX = [y for y in test_y]
    predictions = list()
    for i in range(len(test_y)):
        # make prediction
        predictions.append(historyX[-1])
        # observation
        historyX.append(test_y[i])
    #report performance
    rmse = sqrt(mean_squared_error(test_y, predictions))
    print('Benchmark RMSE test set: %.3f' % rmse)

    ############################################
    # Model

    neurons = config['neurons']
    drop = config['drop']
    nlayers = config['nlayers']
    RNN = LSTM if config['rnn'] == 'LSTM' else GRU

    activation = config['activation']
    activation_r = config['activation_r']

    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], 1), implementation=impl, dropout=drop,
                      activation=activation, recurrent_activation=activation_r))
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], 1), implementation=impl, dropout=drop,
                      activation=activation, recurrent_activation=activation_r, return_sequences=True))
        for i in range(1, nlayers-1):
            model.add(RNN(neurons, dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, dropout=drop, implementation=impl,
                      activation=activation, recurrent_activation=activation_r))
    model.add(Dense(1))

    print('lag: ', lag, 'Neurons: ', neurons, 'Layers: ', nlayers, activation, activation_r)
    print()

    ############################################
    # Training

    optimizer = RMSprop(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])

    model.summary()

    batch_size = config['batch']
    nepochs = config['epochs']

    history = model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=nepochs,
              verbose=verbose, 
	      shuffle=True, 
              validation_data=(val_x, val_y),
	      callbacks=[MeanSquaredErrorHistory(), LossHistory()])

    ############################################
    # Results

    print()
    score = model.evaluate(val_x, val_y,
                           batch_size=batch_size,
                           verbose=0)

    print('MSE Val= ', score)
    print('MSE Val persistence =', mean_squared_error(val_y[1:], val_y[0:-1]))
    print('test loss:', score[0])
    print(history.history.keys())
    #dict_keys(['val_loss', 'val_mean_squared_error', 'loss', 'mean_squared_error'])

    #Accuracy plot
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(history.history['mean_squared_error'])
    ax.plot(history.history['val_mean_squared_error'])
    ax.set_title('Mean squared error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean squared error')
    ax.legend(['train','val'], loc='upper left')
    fig.savefig(ModelName+'_mean_squared_error.pdf')

    #Loss plot
    fig2 = Figure()
    FigureCanvas(fig2)
    ax = fig2.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(['train','val'], loc='upper left')
    fig2.savefig(ModelName+'_loss.pdf')
