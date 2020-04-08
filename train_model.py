import utils.lazy as lazy_import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config

# Lazy loading...
optimizers = lazy_import.lazy_module( 'keras.optimizers' )
Model = lazy_import.lazy_callable( 'keras.models.Model' )
Dense = lazy_import.lazy_callable( 'keras.layers.Dense' )
Input = lazy_import.lazy_callable( 'keras.layers.Input' )
LSTM = lazy_import.lazy_callable( 'keras.layers.LSTM' )
Dropout = lazy_import.lazy_callable( 'keras.layers.Dropout' )
Activation = lazy_import.lazy_callable( 'keras.layers.Activation' )
concatenate = lazy_import.lazy_callable( 'keras.layers.concatenate' )

from utils.file import timestamp_file
from utils.imports.dataset import from_csv


np.random.seed(4)
tf.random.set_seed(4)

# dataset
date_time, ohlcv_histories, technical_indicators, next_day_close_values, unscaled_y, y_normaliser = from_csv( config.data_file )

test_split = 0.9
n = int( ohlcv_histories.shape[0] * test_split )

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_close_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_close_values[n:]
date_time_axis = date_time[n:]

unscaled_y_test = unscaled_y[n:]

print( ohlcv_train.shape )
print( ohlcv_test.shape )
print( date_time_axis.shape )

# model architecture

# define two sets of inputs
lstm_input = Input( shape=( config.history_points, config.num_data_columns ), name='lstm_input' )
dense_input = Input( shape=( technical_indicators.shape[1], ), name='tech_input' )

# the first branch operates on the first input
x = LSTM( 50, name='lstm_0' )( lstm_input )
x = Dropout( 0.2, name='lstm_dropout_0')( x )
lstm_branch = Model( inputs=lstm_input, outputs=x )

# the second branch opreates on the second input
y = Dense( 20, name='tech_dense_0' )( dense_input )
y = Activation( "relu", name='tech_relu_0' )( y )
y = Dropout( 0.2, name='tech_dropout_0' )( y )
technical_indicators_branch = Model( inputs=dense_input, outputs=y )

# combine the output of the two branches
combined = concatenate( [lstm_branch.output, technical_indicators_branch.output], name='concatenate' )

z = Dense( 64, activation="sigmoid", name='dense_pooling' )( combined )
z = Dense( 1, activation="linear", name='dense_out' )( z )

# our model will accept the inputs of the two branches and
# then output a single value
model = Model( inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z )
adam = optimizers.Adam( lr=0.0005 )
model.compile( optimizer=adam, loss='mse' )
# x = new input data
# y = predicted data
model.fit( x=[ohlcv_train, tech_ind_train], y=y_train, batch_size=32, epochs=50, shuffle=True, validation_split=0.1 )


# evaluation
y_test_predicted = model.predict( [ohlcv_test, tech_ind_test] )
y_test_predicted = y_normaliser.inverse_transform( y_test_predicted )
y_predicted = model.predict( [ohlcv_histories, technical_indicators] )
y_predicted = y_normaliser.inverse_transform( y_predicted )

assert unscaled_y_test.shape == y_test_predicted.shape == date_time_axis.shape

real_mse = np.mean( np.square( unscaled_y_test - y_test_predicted ) )
scaled_mse = real_mse / ( np.max( unscaled_y_test ) - np.min( unscaled_y_test ) ) * 100
print( scaled_mse )

start = 0
end = -1

real = plt.plot( date_time_axis[start:end], unscaled_y_test[start:end], label='real' )
pred = plt.plot( date_time_axis[start:end], y_test_predicted[start:end], label='predicted' )

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.gcf().set_size_inches( 22, 15, forward=True )
plt.legend( ['Real', 'Predicted'] )
plt.savefig( timestamp_file( config.train_analysis ) )
plt.show()

model.save( config.model_file )
