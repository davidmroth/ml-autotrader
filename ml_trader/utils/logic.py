import numpy as np

import ml_trader.config as config


def do_trade( model, trade_data, y_normaliser ):
    buys = []
    sells = []
    thresh = config.trade_threshold #in dollars?
    predicted_price_yhat = np.array( [None] )

    for ohlcv, ind, date in trade_data:
        normalised_price_today = np.array( [[ohlcv[-1][0]]] )

        price_today = y_normaliser.inverse_transform( normalised_price_today )

        predicted_price_tomorrow = np.squeeze( y_normaliser.inverse_transform( model.predict( [[ohlcv], [ind]] ) ) )

        # For plotting
        predicted_price_yhat = np.append( predicted_price_yhat, predicted_price_tomorrow )

        delta = predicted_price_tomorrow - price_today

        # Buy / Sell Logic
        if delta > thresh:
            buys.append( ( date, price_today[0][0] ) )

        elif delta < -thresh:
            sells.append( ( date, price_today[0][0] ) )

        #print( "X %s, ind: %s " % ( x, ind ) )

    return ( predicted_price_yhat, buys, sells )
