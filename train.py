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
# Setup
preprocess = Preprocess( 0.9 )
# Training data
ohlcv_train, tech_ind_train, y_train, y_train_dates = preprocess.get_training_data()
# Test data
ohlcv_test, tech_ind_test, y_test, y_test_dates = preprocess.get_test_data()
# Other
unscaled_y_test = preprocess.get_unscaled_data()


'''
Train model
'''
technical_model = Technical_Model( preprocess.get_y_normalizer() ) # Instantiate class
technical_model.build() # Build model
evalutation = technical_model.train( [ohlcv_train, tech_ind_train], y_train, [ohlcv_test, tech_ind_test], y_test ) # Train model
#evalutation = technical_model.optimized_training( [ohlcv_train, tech_ind_train], y_train, [ohlcv_test, tech_ind_test], y_test ) # Train model
technical_model.save() # Save trained model for later use


'''
Evaluate model
'''
y_train_predicted = technical_model.predict( [ohlcv_train, tech_ind_train] )
y_test_predicted = technical_model.predict( [ohlcv_test, tech_ind_test] )

# Check
assert unscaled_y_test.shape == y_test_predicted.shape


'''
Score model
'''

'''
What does the Mean Squared Error Tell You?

The smaller the means squared error, the closer you are to finding the line of
best fit. Depending on your data, it may be impossible to get a very small
value for the mean squared error. For example, the above data is scattered
wildly around the regression line, so 6.08 is as good as it gets (and is in
fact, the line of best fit).
'''
mse = technical_model.mean_squaured_error( unscaled_y_test, y_test_predicted )
print( "Mean Squared Error: %.4f" %  mse )


'''
Plot
'''
plt = Plot( 'Training', start=0, end=-1, xlabel='Date', ylabel='Stock Price' )
plt.title( 'Training Result' )
plt.graph( x_axis=y_test_dates, y_axis=unscaled_y_test, label='Real' )
plt.graph( x_axis=y_test_dates, y_axis=y_test_predicted, label='Predicted' )
plt.add_note(
    (
        r'Date: %s' % ( time.strftime( "%m/%d/%Y %H:%M:%S" ) ),
        r'Symbol: %s' % ( config.stock_symbol, ),
        r'MSE: %.2f' % ( mse, ),
        r'Epochs: %d' % ( config.epochs, ),
        r'Data Points: %d' % ( unscaled_y_test.shape[0], ),
        r'History Points: %d' % ( config.history_points, )
    )
)
plt.create()

#TODO: Put accuracy on a seperate scale shared by the same X axis
acc, loss = technical_model.score( [ohlcv_test, tech_ind_test], y_test )
print( "Evalutation: Loss: %.4f, Accuracy: %.4f" % ( acc, loss ) )

plt = Plot( 'Training_loss', start=0, end=-1, xlabel='Epochs', ylabel='Loss' )
plt.title( 'Model Training vs. Validation Loss' )
plt.graph( y_axis=evalutation.history['loss'], label='Train Loss' )
plt.graph( y_axis=evalutation.history['val_loss'], label='Test Loss' )
plt.graph( y_axis=evalutation.history['accuracy'], label='Train accuracy' )
plt.graph( y_axis=evalutation.history['val_accuracy'], label='Test accuracy' )
plt.create()
