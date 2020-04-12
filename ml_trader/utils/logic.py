import datetime
import numpy as np

import ml_trader.config as config
import ml_trader.utils as utils
import ml_trader.utils.data.meta as meta

def get_insight( date, price_today, predicted_price_tomorrow ):
    # I can only predict the next day, but I need all of today's stats for today, so I
    # can only work at the end of the day. That means I have to buy stock
    # low or high on the following day based on the output of the model

    # Input the ochlv (for the previous 50 days?)
    # Output the following day's close price

    change = False
    delta = np.round( predicted_price_tomorrow - price_today, 2 ).item()
    day = utils.convert_to_datetime( date )

    #TODO: get actual next trading day
    #next_day = day + datetime.timedelta( days=1 )
    #TODO: Add trading after a holidy indictor for the model, so it will learn
    # to recognize trading patterns after a holiday (?)
    next_day = 'the next trading day'

    percent_change = abs( float( '{:.2f}'.format( ( 100 * ( 1 - ( predicted_price_tomorrow / price_today ) ) ).item() ) ) )
    decrease_increase = 'decrease' if delta < 0 else 'increase'
    change_summary_text = 'which will represent a ${0:.2f} {1} of {2:.2f}%'

    if not ( delta > 0 or delta < 0 ):
        change_summary_text = 'which will represents no change'

    if percent_change < 1:
        change_summary_text = 'which will represent a ${0:.2f} {1} of less than 1%'

    price_change_summary = change_summary_text.format(
        abs( delta ),
        decrease_increase,
        percent_change
    )

    print(
        '''
        Using data points for: {:%b %d, %Y}, I predict that the stock will close on {} at
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

def do_trade( model, packed_trade_data, y_normaliser ):
    # Initialize
    start = 0
    end = -1
    buys = []
    sells = []
    thresh = config.trade_threshold #in dollars?
    predicted_price_yhat = np.array( [None] )

    # Trasnform data
    ohlcv_test, tech_ind_test, y_test_dates = packed_trade_data

    #TODO: Do I need start and end? If 0 ='s the begining the array, and -1
    # ='s the 'end' of array, which whould mean the whole array
    trade_data = zip( ohlcv_test[start: end], tech_ind_test[start: end], y_test_dates[start: end] )

    for ohlcv, ind, date in trade_data:
        # Get the last 'close' price in history
        normalised_price_today = [[ohlcv[-1][meta.column_index['close']]]]

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
        get_insight( date, price_today, predicted_price_tomorrow )

    return ( predicted_price_yhat, buys, sells )
