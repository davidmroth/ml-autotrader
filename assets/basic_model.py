import keras
import numpy as np
import pandas as pd #debug
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation

from util import csv_to_dataset, history_points


np.random.seed(4)
tf.random.set_seed(4)

num_data_columns = 6

# dataset
ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset( 'MSFT_daily.csv' )
print( ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser )

test_split = 0.9 #Get 90 of the data as training samples
n = int( ohlcv_histories.shape[0] * test_split )

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print( ohlcv_train.shape )
print( ohlcv_test.shape )


# model architecture
lstm_input = Input( shape=( history_points, num_data_columns ), name='lstm_input' )
x = LSTM( 50, name='lstm_0' )( lstm_input )
x = Dropout( 0.2, name='lstm_dropout_0' )( x )
x = Dense( 64, name='dense_0' )( x )
x = Activation( 'sigmoid', name='sigmoid_0' )( x )
x = Dense( 1, name='dense_1' )( x )
output = Activation( 'linear', name='linear_output' )( x )

model = Model( inputs=lstm_input, outputs=output )
adam = optimizers.Adam( lr=0.0005 )
model.compile( optimizer=adam, loss='mse' )
model.fit( x=ohlcv_train, y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1 )


# evaluation
y_test_predicted = model.predict( ohlcv_test )
y_test_predicted = y_normaliser.inverse_transform( y_test_predicted )
y_predicted = model.predict( ohlcv_histories )
y_predicted = y_normaliser.inverse_transform( y_predicted )

assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean( np.square( unscaled_y_test - y_test_predicted ) )
scaled_mse = real_mse / ( np.max(unscaled_y_test ) - np.min( unscaled_y_test ) ) * 100
print( scaled_mse )

plt.gcf().set_size_inches( 22, 15, forward=True )

start = 0
end = -1

real = plt.plot( unscaled_y_test[start:end], label='real' )
pred = plt.plot( y_test_predicted[start:end], label='predicted' )

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend( ['Real', 'Predicted'] )
plt.savefig('books_read.png')
plt.show()


model.save( f'basic_model.h5' )
