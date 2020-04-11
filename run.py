import time
import argparse
import datetime
import numpy as np

import ml_trader.utils as utils
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


    ohlcv = np.array( [ohlcv] )
    tech_ind = np.array( [tech_ind] )
    y_date = utils.convert_to_datetime( y_date )


    '''
    Run model
    '''
    technical_model = Technical_Model( y_normaliser ).load() # Load model
    y_price_predicted = technical_model.predict( [ohlcv, tech_ind] )
    
    print(
        'The price predicted is {:.2f} which is the day after {}'.format(
            y_price_predicted[0][0],
            y_date
        )
    )


def valid_date( s ):
    type( s )
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
        type=valid_date,
        help="The day you predict the stock price; defaults to the next business day"
    )

    namespace = parser.parse_args()
    predict( **vars( namespace ) )
