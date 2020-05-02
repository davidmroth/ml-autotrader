import datetime
import numpy as np

from ml_trader.config import Config as config
from ml_trader.utils.analysis.insight import Insight


def watch_last( iterable ):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False).
    """
    # Get an iterator and pull the first value.
    it = iter( iterable )
    last = next( it )

    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield *last, False
        last = val

    # Report the last value.
    yield *last, True

def do_trade( model, packed_trade_data, normaliser ):
    # Initialize
    track_metrics = []
    buys = []
    sells = []
    thresh = config.trade_threshold #in dollars?
    predicted_price_yhat = np.array( [None] )

    # Trasnform data
    ohlcv_test, tech_ind_test, y_test_dates, y_test = packed_trade_data

    #TODO: Merge date column to np array and then I can do:
    insight = Insight( normaliser[config.label_column].inverse_transform( y_test ), y_test_dates )

    #TODO: Do I need start and end? If 0 ='s the begining the array, and -1
    # ='s the 'end' of array, which whould mean the whole array
    trade_data = zip( ohlcv_test, tech_ind_test, y_test_dates, y_test )

    for ohlcv, ind, date, y_test, is_last in watch_last( trade_data ):
        # Get actual price (not normalized)
        price_today = normaliser[config.label_column].inverse_transform( [y_test] )

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
        #predicted_price_yhat = np.append( predicted_price_yhat, predicted_price_tomorrow )

        # Print insight
        insight.get_trade_insight( date, price_today, predicted_price_tomorrow, is_last, track_metrics )

    #TODO: Track average error in prediction in dollars missed by
    print( "\n\nSummary of up/down predictions: \n\tTotal: {:d}\n\tCorrect: {:d}\n\tPercent correct: {:.2f}%\n".format(
        len( track_metrics ),
        np.sum( track_metrics ),
        np.sum( track_metrics ) / len( track_metrics ) * 100 )
    )
    return ( buys, sells )
