import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ml_trader.config as config
import ml_trader.utils.data.meta as meta

from tensorboard import program
from ml_trader.models.technical import Technical_Model
from ml_trader.utils.analysis.plot import Plot
from ml_trader.utils.data import Preprocess


np.random.seed( 4 )
tf.random.set_seed( 4 )


if __name__ == '__main__':
    # Start tensorboard
    log_dir='/tmp'
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--host=0.0.0.0', '--port=8080'])
    url = tb.launch()

    '''
    Retreive & preprocess data for ML model
    '''
    # Setup
    preprocess = Preprocess( 0.9 )

    # Training data
    ohlcv_train, tech_ind_train, y_train, y_train_dates = preprocess.get_training_data()

    # Test data
    ohlcv_test, tech_ind_test, y_test, y_test_dates = preprocess.get_test_data()

    # Get data normalizer
    scalers = preprocess.get_scalers()

    print( ohlcv_train.shape )
    print( ohlcv_test.shape )


    '''
    Train model
    '''
    technical_model = Technical_Model( scalers ) # Instantiate class
    technical_model.build() # Build model
    evalutation = technical_model.train( [ohlcv_train, tech_ind_train], y_train, [ohlcv_test, tech_ind_test], y_test ) # Train model
    #evalutation = technical_model.optimized_training( [ohlcv_train, tech_ind_train], y_train, [ohlcv_test, tech_ind_test], y_test ) # Train model

    '''
    Evaluate model
    '''
    y_train_predicted = technical_model.predict( [ohlcv_train, tech_ind_train] )
    y_test_predicted = technical_model.predict( [ohlcv_test, tech_ind_test] )

    #TODO: Put accuracy on a seperate scale shared by the same X axis
    loss, acc, mae = technical_model.score( [ohlcv_test, tech_ind_test], y_test_predicted )

    # Check
    assert y_test.shape == y_test_predicted.shape


    '''
    Score model
    '''
    mse = technical_model.mean_squaured_error( y_test, y_test_predicted )
    rmse = technical_model.root_mean_squared_error( y_test, y_test_predicted )
    print( "\n\nEvalutation: \n \
        \tLoss: %.4f\n \
        \tAccuracy: %.4f\n \
        \tMean Absolute Error: %.2f\n \
        \tMean Squared Error: %.2f\n \
        \tRoot Mean Squared Error %.2f" % ( loss, acc, mae, mse, rmse ) )


    '''
    Plot
    '''
    plt = Plot( 'Training', xlabel='Date', ylabel='Stock Price', scalers=scalers )
    plt.title( 'Training Result' )
    plt.graph( x_axis=y_test_dates, y_axis=y_test, label='Real', scale=True )
    plt.graph( x_axis=y_test_dates, y_axis=y_test_predicted, label='Predicted' )
    plt.add_note(
        (
            r'Date: %s' % ( time.strftime( "%m/%d/%Y %H:%M:%S" ) ),
            r'Symbol: %s' % ( config.stock_symbol, ),
            r'MSE: %.2f' % ( mse, ),
            r'Epochs: %d' % ( config.epochs, ),
            r'Data Points: %d' % ( y_test.shape[0], ),
            r'History Points: %d' % ( config.history_points, )
        )
    )
    plt.create()

    plt = Plot( 'Training_loss', xlabel='Epochs', ylabel='Loss' )
    plt.title( 'Model Training vs. Validation Loss' )
    plt.graph( y_axis=evalutation.history['loss'], label='Train Loss' )
    plt.graph( y_axis=evalutation.history['val_loss'], label='Test Loss' )
    plt.graph( y_axis=evalutation.history['accuracy'], label='Train accuracy' )
    plt.graph( y_axis=evalutation.history['val_accuracy'], label='Test accuracy' )
    plt.create()
