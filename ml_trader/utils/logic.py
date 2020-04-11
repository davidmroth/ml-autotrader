import datetime
import numpy as np

import ml_trader.config as config
import ml_trader.utils as utils
import ml_trader.utils.data.meta as meta

def get_insight( date, price_today, predicted_price_tomorrow, delta ):
    # I can only predict the next day, but I need all of today's stats for today, so I
    # can only work at the end of the day. That means I have to buy stock
    # low or high on the following day based on the output of the model

    # Input the ochlv (for the previous 50 days?)
    # Output the following day's close price

    change = False
    delta = np.round( predicted_price_tomorrow - price_today, 2 ).item()
    day = utils.convert_to_datetime( date )
    next_day = day + datetime.timedelta( days=1 )

    percent_change = abs( 1 - float( '{:.2f}'.format( ( predicted_price_tomorrow / price_today ).item() ) ) )
    decrease_increase = 'decrease' if delta < 0 else 'increase'

    if delta > 0 or delta < 0:
        change = True

    if change:
        price_change_summary = 'which will represent a ${:.2f} {} of {:.2f}%'.format(
            delta,
            decrease_increase,
            percent_change
        )
    else:
        price_change_summary = 'which will represents no change'.format(
            delta,
            decrease_increase,
            percent_change
        )

    print(
        '''
        On, {:%b %d, %Y} I predict that the stock will close on {:%b %d, %Y} at
        '${:.2f}', {} over the stock
        price close today at '${:.2f}'.
        '''.format(
            day,
            next_day,
            predicted_price_tomorrow.item(),
            price_change_summary,
            price_today[0][0].item(),
        )
    )

def do_trade( model, trade_data, y_normaliser ):
    # Initialize
    start = 0
    end = -1
    buys = []
    sells = []
    thresh = config.trade_threshold #in dollars?
    predicted_price_yhat = np.array( [None] )

    # Trasnform data
    ohlcv_test, tech_ind_test, y_test_dates = trade_data
    trade_data = zip( ohlcv_test[start: end], tech_ind_test[start: end], y_test_dates[start: end] )


    for ohlcv, ind, date in trade_data:
        # Get the last 'close' price in history
        normalised_price_today = np.array( [[ohlcv[-1][meta.column_index['close']]]] )

        # Get actual price (not normalized)
        price_today = y_normaliser.inverse_transform( normalised_price_today )

        # Get predicted 'close' price for the next day
        predicted_price_tomorrow = np.squeeze( model.predict( [[ohlcv], [ind]] ) )

        # Get price difference
        delta = predicted_price_tomorrow - price_today

        # Buy / Sell Logic
        if delta > thresh:
            buys.append( ( date, price_today[0][0] ) )

        elif delta < -thresh:
            sells.append( ( date, price_today[0][0] ) )

        #print( "X %s, ind: %s " % ( x, ind ) )

        # For plotting: append predicted prices into an array
        predicted_price_yhat = np.append( predicted_price_yhat, predicted_price_tomorrow )

        # Print insight
        get_insight( date, price_today, predicted_price_tomorrow, delta )

    return ( predicted_price_yhat, buys, sells )
