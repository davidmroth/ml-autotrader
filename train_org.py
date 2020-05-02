import os
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras.callbacks import TensorBoard
from keras import optimizers
import numpy as np
from tensorboard import program
import datetime

import ml_trader.utils.file as file

from ml_trader.config import Config as config
from ml_trader.utils.data import Preprocess
from ml_trader.utils.orig import csv_to_dataset


np.random.seed( 4 )
tf.random.set_seed( 4 )


if __name__ == '__main__':
    # Start tensorboard
    log_dir='/tmp'
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--host=0.0.0.0', '--port=8080'])
    url = tb.launch()


    '''
    new
    '''
    preprocess = Preprocess( 0.9 )

    # Training data
    ohlcv_train, tech_ind_train, y_train, y_train_dates = preprocess.get_training_data()

    # Test data
    ohlcv_test, tech_ind_test, y_test, y_test_dates = preprocess.get_test_data()

    # All data
    ohlcv_histories, technical_indicators, unscaled_y, dates = preprocess.get_all_data()

    # Get data normalizer
    scalers = preprocess.get_scalers()

    y_normaliser = scalers[config.label_column]
    unscaled_y = y_normaliser.inverse_transform( unscaled_y )

    print(
        #"\n\nHistories: ", ohlcv_histories,
        #'\n\nTech_Ind:', technical_indicators,
        '\n\nY:', unscaled_y[0]
    )

    '''
    old
    '''
    #'''
    # dataset
    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset( 'data/stock/SPY_daily.csv' )

    print( '#########################################\n\n\n' )
    print(
        #"\n\nHistories: ", ohlcv_histories,
        #'\n\nTech_Ind:', technical_indicators,
        '\n\nY:', unscaled_y[0]
    )

    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    tech_ind_train = technical_indicators[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]

    #exit()
    #'''




    print( ohlcv_train.shape )
    print( ohlcv_test.shape )
    print( tech_ind_train.shape[1] )


    # model architecture

    # define two sets of inputs
    lstm_input = Input(shape=(config.history_points, 5), name='lstm_input')
    dense_input = Input(shape=(tech_ind_train.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')

    file.create_path_if_needed( config.model_filepath )
    dir = os.path.dirname( config.model_filepath )
    fname = os.path.basename( config.model_filepath )
    log_dir = "/tmp/tensorboard/" + datetime.datetime.now().strftime( "%Y%m%d-%H%M%S" )
    tensorboard = TensorBoard( log_dir=os.path.join( log_dir, 'original_model' ) )

    model.fit(x=[ohlcv_train, tech_ind_train], callbacks=[tensorboard], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1)


    # evaluation

    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

    y_predicted = model.predict([ohlcv_histories, technical_indicators])
    y_predicted = y_normaliser.inverse_transform(y_predicted)

    unscaled_y_test = y_normaliser.inverse_transform(y_test)
    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
    print(scaled_mse)
