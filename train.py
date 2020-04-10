import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ml_trader.config as config

from ml_trader.models.technical import Technical_Model
from ml_trader.utils.analysis.plot import Plot
from ml_trader.utils.data import Preprocess


np.random.seed( 4 )
tf.random.set_seed( 4 )


'''
Retreive & preprocess data for ML model
'''
preprocess = Preprocess( 0.9 )
# Training data
ohlcv_train, tech_ind_train, y_train, y_train_dates = preprocess.get_training_data()
# Test data
ohlcv_test, tech_ind_test, y_test, y_test_dates = preprocess.get_test_data()
# Other
unscaled_y_test = preprocess.get_unscaled_data()
y_normaliser = preprocess.get_y_normalizer()


'''
Train model
'''
technical_model = Technical_Model( y_normaliser ) # Instantiate class
technical_model.build( tech_ind_train.shape[1] ) # Build model
history = technical_model.train( [ohlcv_train, tech_ind_train], y_train, [ohlcv_test, tech_ind_test], y_test ) # Train model
technical_model.save() # Save trained model for later use


'''
Evaluate model
'''
y_train_predicted = technical_model.predict( [ohlcv_train, tech_ind_train] )
y_test_predicted = technical_model.predict( [ohlcv_test, tech_ind_test] )

assert unscaled_y_test.shape == y_test_predicted.shape


'''
Analysis & Scoring
'''

'''
Mean Squared Error Definition

The mean squared error tells you how close a regression line is to a set of
points. It does this by taking the distances from the points to the regression
line (these distances are the “errors”) and squaring them. The squaring is
necessary to remove any negative signs. It also gives more weight to larger
differences. It’s called the mean squared error as you’re finding the average
of a set of errors.
'''

real_mse = np.mean( np.square( unscaled_y_test - y_test_predicted ) )
scaled_mse = real_mse / ( np.max( unscaled_y_test ) - np.min( unscaled_y_test ) ) * 100
print( "Mean Squared Error:", scaled_mse )

'''
What does the Mean Squared Error Tell You?

The smaller the means squared error, the closer you are to finding the line of
best fit. Depending on your data, it may be impossible to get a very small
value for the mean squared error. For example, the above data is scattered
wildly around the regression line, so 6.08 is as good as it gets (and is in
fact, the line of best fit).
'''


'''
Plot
'''
plt = Plot( 'Training', start=0, end=-1, xlabel='Date', ylabel='Stock Price' )
plt.graph( x_axis=y_test_dates, y_axis=unscaled_y_test, label='Real' )
plt.graph( x_axis=y_test_dates, y_axis=y_test_predicted, label='Predicted' )
plt.add_note(
    (
        r'Date: %s' % ( time.strftime( "%m/%d/%Y %H:%M:%S" ) ),
        r'Symbol: %s' % ( config.stock_symbol, ),
        r'MSE: %.2f' % ( scaled_mse, ),
        r'Epochs: %d' % ( config.epochs, ),
        r'History Points: %d' % ( config.history_points, )
    )
)
plt.create()

plt = Plot( 'Training_loss', start=0, end=-1, xlabel='Epochs', ylabel='Loss' )
plt.graph( y_axis=history.history['loss'], label='Train Loss' )
plt.graph( y_axis=history.history['val_loss'], label='Test Loss' )
plt.create()
