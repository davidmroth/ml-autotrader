import time
import numpy as np

import ml_trader.config as config

from ml_trader.models.technical import Technical_Model
from ml_trader.utils.analysis.plot import Plot
from ml_trader.utils import compute
from ml_trader.utils.file import timestamp
from ml_trader.utils.data import Preprocess


preprocess = Preprocess( 0.9 )
unscaled_y_test, date_time_axis = preprocess.get_unscaled_data()
ohlcv_train, tech_ind_train, y_train = preprocess.get_training_data()
ohlcv_test, tech_ind_test, y_test = preprocess.get_test_data()
y_normaliser = preprocess.get_y_normalizer()

# Run model
technical_model = Technical_Model( y_normaliser ).load() # Load model
y_test_predicted = technical_model.predict( [ohlcv_test, tech_ind_test] )


# Analysis and plotting
buys = []
sells = []
thresh = config.trade_threshold #in dollars?
x = -1
start = 0
end = -1

for ohlcv, ind in zip( ohlcv_test[start: end], tech_ind_test[start: end] ):
    normalised_price_today = ohlcv[-1][0]
    normalised_price_today = np.array( [[normalised_price_today]] )
    price_today = y_normaliser.inverse_transform( normalised_price_today )

    predicted_price_tomorrow = np.squeeze( y_normaliser.inverse_transform( technical_model.predict( [[ohlcv], [ind]] ) ) )
    delta = predicted_price_tomorrow - price_today

    # Buy / Sell Logic
    if delta > thresh:
        buys.append( ( x, price_today[0][0] ) )

    elif delta < -thresh:
        sells.append( ( x, price_today[0][0] ) )

    #print( "X %s, ind: %s " % ( x, ind ) )
    x += 1

print( "Buys: %d" % len( buys ) )
print( "Sells: %d" % len( sells ) )

# we create new lists so we dont modify the original
compute.earnings( [b for b in buys], [s for s in sells] )

# Score
real_mse = np.mean( np.square( unscaled_y_test - y_test_predicted ) )
scaled_mse = real_mse / ( np.max( unscaled_y_test ) - np.min( unscaled_y_test ) ) * 100
print( "Mean Squared Error (MSE): %.2f" % scaled_mse )

# Plot
plt = Plot( 'Simulation', start=0, end=-1, legend=['Real', 'Predicted', 'Buy', 'Sell'] )
plt.graph( x_axis=date_time_axis, y_axis=unscaled_y_test, label='Real' )
plt.graph( x_axis=date_time_axis, y_axis=y_test_predicted, label='Predicted' )
plt.add_note(
    (
        r'Date: %s' % ( time.strftime( "%m/%d/%Y %H:%M:%S" ) ),
        r'Symbol: %s' % ( config.stock_symbol, ),
        r'MSE: %.2f' % ( scaled_mse, ),
        r'Epochs: %d' % ( config.epochs, ),
        r'History Points: %d' % (config.history_points, )
    )
)

'''
if len( buys ) > 0:
    plt.plot_buys_and_sells( x_axis=buys, x_index=0, y_axis=buys, y_index=1 , c='#00ff00', s=50 )

if len( sells ) > 0:
    plt.plot_buys_and_sells( x_axis=sells, x_index=0, y_axis=sells, y_index=1 , c='#ff0000', s=50 )
'''

plt.create()
