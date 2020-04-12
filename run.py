import time
import argparse
import datetime
import numpy as np

import ml_trader.utils as utils
import ml_trader.utils.data.meta as meta
import ml_trader.config as config
import ml_trader.utils.logic as trade_logic

from ml_trader.models.technical import Technical_Model
from ml_trader.utils.analysis.plot import Plot
from ml_trader.utils.file import timestamp
from ml_trader.utils.data import Preprocess


def predict( date ):
    date = datetime.datetime.strptime( date, '%m/%d/%Y' )

    '''
    Retreive & preprocess data for ML model
    '''
    preprocess = Preprocess()
    ohlcv, tech_ind, y, y_date = preprocess.get_history_for_date( date )
    y_normaliser = preprocess.get_y_normalizer()


    '''
    Run model
    '''
    technical_model = Technical_Model( y_normaliser ).load() # Load model
    y_price_predicted = technical_model.predict( [[ohlcv], [tech_ind]] )

    #BUG: Last column is not the current day's metrics.
    #TODO: Need to figure out how to get the current day's metrics(?)
    price_today = y_normaliser.inverse_transform( np.array( [[ohlcv[-1][meta.meta.label_column]]] ) )
    trade_logic.get_insight( y_date, price_today, y_price_predicted )

def check_valid_date( s ):
    try:
        datetime.datetime.strptime( s, "%m/%d/%Y" )
        return s

    except ValueError:
        msg = "Not a valid date: '{0}'. Should be m/d/Y, e.g., 03/02/2020".format( s )
        raise argparse.ArgumentTypeError( msg )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'date',
        type=check_valid_date,
        help="The day you predict the stock price; defaults to the next business day"
    )

    namespace = parser.parse_args()
    predict( **vars( namespace ) )
