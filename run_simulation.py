import time
import numpy as np

import ml_trader.config as config
import ml_trader.utils.data.meta as meta
import ml_trader.utils.logic as trade_logic

from ml_trader.models.technical import Technical_Model
from ml_trader.utils import compute
from ml_trader.utils.analysis.plot import Plot
from ml_trader.utils.file import timestamp
from ml_trader.utils.data import Preprocess
import ml_trader.utils as utils


'''
Retreive & preprocess data for ML model
'''
preprocess = Preprocess( 0.9 )

# Training data
ohlcv_train, tech_ind_train, y_train, y_train_dates = preprocess.get_training_data()

# Test data
ohlcv_test, tech_ind_test, y_test, y_test_dates = preprocess.get_test_data()

# Other
normaliser = preprocess.get_scalers()


'''
Run model
'''
technical_model = Technical_Model( normaliser ).load() # Load model
y_test_predicted = technical_model.predict( [ohlcv_test, tech_ind_test] )


'''
Buy / Sell Trade Logic
'''
trade_data = ( ohlcv_test, tech_ind_test, y_test_dates, y_test )
buys, sells = trade_logic.do_trade( technical_model, trade_data, normaliser )


'''
Analysis & Scoring
'''
print( "Buys: %d" % len( buys ) )
print( "Sells: %d" % len( sells ) )

# We create new lists so we dont modify the original
compute.earnings( buys, sells )

# TODO: Move to model scoring
mse = technical_model.mean_squaured_error( y_test, y_test_predicted )
print( "Mean Squared Error (MSE): %.2f" % mse )
print( 'Num of predictions ran: %d' % ( y_test_predicted.shape[0], ) )


'''
Plot
'''
unscaled_y_test = normaliser[meta.label_column].inverse_transform( y_test )
plt = Plot( 'Simulation', start=0, end=-1, legend=['Real', 'Predicted', 'Buy', 'Sell'] )
plt.graph( x_axis=y_test_dates, y_axis=unscaled_y_test, label='Real' )
plt.graph( x_axis=y_test_dates, y_axis=y_test_predicted, label='Predicted' )
plt.add_note(
    (
        r'Date: %s' % ( time.strftime( "%m/%d/%Y %H:%M:%S" ) ),
        r'Symbol: %s' % ( config.stock_symbol, ),
        r'MSE: %.2f' % ( mse, ),
        r'Epochs: %d' % ( config.epochs, ),
        r'History Points: %d' % (config.history_points, ),
        r'Total days: %d' % ( y_test_predicted.shape[0], )
    )
)

if len( buys ) > 0:
    plt.plot_buys_and_sells( x_axis=buys, x_index=0, y_axis=buys, y_index=1 , c='#00ff00', s=50 )

if len( sells ) > 0:
    plt.plot_buys_and_sells( x_axis=sells, x_index=0, y_axis=sells, y_index=1 , c='#ff0000', s=50 )

plt.create()
