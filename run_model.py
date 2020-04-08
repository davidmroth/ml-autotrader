import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

from utils import compute
from utils.imports.dataset import from_csv
from config import history_points


model = load_model( 'models/technical_model.h5' )

data_time, ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = from_csv( 'data/MSFT_daily.csv' )

test_split = 0.9
n = int( ohlcv_histories.shape[0] * test_split )

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

y_test_predicted = model.predict( [ohlcv_test, tech_ind_test] )
y_test_predicted = y_normaliser.inverse_transform( y_test_predicted )

buys = []
sells = []
thresh = 0.1 #in dollars?
start = 0
end = -1
x = -1

for ohlcv, ind in zip( ohlcv_test[start: end], tech_ind_test[start: end] ):
    normalised_price_today = ohlcv[-1][0]
    normalised_price_today = np.array( [[normalised_price_today]] )
    price_today = y_normaliser.inverse_transform( normalised_price_today )
    predicted_price_tomorrow = np.squeeze( y_normaliser.inverse_transform( model.predict( [[ohlcv], [ind]] ) ) )
    delta = predicted_price_tomorrow - price_today

    if delta > thresh:
        buys.append( ( x, price_today[0][0] ) )

    elif delta < -thresh:
        sells.append( ( x, price_today[0][0] ) )

    x += 1

print( f"buys: {len( buys )}" )
print( f"sells: {len( sells )}" )

# we create new lists so we dont modify the original
compute.earnings( [b for b in buys], [s for s in sells] )

plt.gcf().set_size_inches( 22, 15, forward=True )

real = plt.plot( unscaled_y_test[start:end], label='real' )
pred = plt.plot( y_test_predicted[start:end], label='predicted' )

if len( buys ) > 0:
    plt.scatter( list( list( zip( *buys ) )[0] ), list( list( zip( *buys ) )[1] ), c='#00ff00', s=50 )

if len( sells ) > 0:
    plt.scatter( list( list( zip( *sells ) )[0] ), list( list( zip( *sells ) )[1] ), c='#ff0000', s=50 )

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend( ['Real', 'Predicted', 'Buy', 'Sell'] )
plt.savefig('analysis/predict_out.png')
plt.show()
