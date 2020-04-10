import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ml_trader.config as config

from ml_trader.models.technical import Technical_Model
from ml_trader.utils.analysis.plot import Plot
from ml_trader.utils import compute
from ml_trader.utils.file import timestamp
from ml_trader.utils.imports.dataset import from_csv


date_time, ohlcv_histories, technical_indicators, next_day_close_values, \
unscaled_y, y_normaliser = from_csv()

# output
print( "date_time:", pd.DataFrame( date_time ).head() )
print( "ohlcv_histories:", ohlcv_histories )
exit()

test_split = 0.9
n = int( ohlcv_histories.shape[0] * test_split )

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_close_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_close_values[n:]

unscaled_y_test = unscaled_y[n:]
date_time_axis = date_time[n:]


# Run model
technical_model = Technical_Model( y_normaliser ).load() # Load model
y_test_predicted = technical_model.predict( [ohlcv_test, tech_ind_test] )


# Analysis and plotting
buys = []
sells = []
thresh = 0.1 #in dollars?
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

print( f"buys: {len( buys )}" )
print( f"sells: {len( sells )}" )

# we create new lists so we dont modify the original
compute.earnings( [b for b in buys], [s for s in sells] )


# Plot
plt = Plot( 'Simulation', start=0, end=-1, legend=['Real', 'Predicted', 'Buy', 'Sell'] )
plt.graph( x_axis=date_time_axis, y_axis=unscaled_y_test, label='Real' )
plt.graph( x_axis=date_time_axis, y_axis=y_test_predicted, label='Predicted' )
plt.add_note( "Text" )

'''
if len( buys ) > 0:
    plt.plot_buys_and_sells( x_axis=buys, x_index=0, y_axis=buys, y_index=1 , c='#00ff00', s=50 )

if len( sells ) > 0:
    plt.plot_buys_and_sells( x_axis=sells, x_index=0, y_axis=sells, y_index=1 , c='#ff0000', s=50 )
'''

plt.create()
