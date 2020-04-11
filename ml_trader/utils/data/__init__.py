import pandas as pd
import numpy as np
import datetime

import ml_trader.config as config
import ml_trader.utils as utils
import ml_trader.utils.file as file
import ml_trader.utils.data.meta as meta
import ml_trader.utils.stock.indicators as stock_indicators
import ml_trader.utils.data.imports.get as get

from pprint import pprint
from sklearn import preprocessing

from ml_trader.utils.compute import earnings


def prepare( data ):
    history_points = config.history_points

    # The first day of trading that stock often looked anomalous due to the massively high volume (IPO).
    # This inflated max volume value also affected how other volume values in the dataset were scaled when normalising the data,
    # so we drop the oldest data point out of every set)

    # Transform data
    data.sort_values( 'date', inplace=True, ascending=True )
    data['date'] = pd.to_datetime( data['date'] )
    data['date'] = data['date'].apply( utils.convert_to_timestamp )
    data = data.drop( 0, axis=0 ) # Drop one day (oldest)
    data = data.values


    # Normalise the data — scale it between 0 and 1 — to improve how quickly our network converges
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform( data )


    # using the last {history_points} open close high low volume data points, predict the next  value
    ohlcv_histories_normalised       = np.array( [data_normalised[i:i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    next_day_close_values_normalised = np.array( [data_normalised[:, meta.column_index['close']][i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    next_day_close_values_normalised = np.expand_dims( next_day_close_values_normalised, -1 )

    next_day_close_values = np.array( [data[:, meta.column_index['close']][i + history_points].copy() for i in range( len( data ) - history_points)] )
    next_day_close_values = np.expand_dims( next_day_close_values, -1 )

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit( next_day_close_values )

    # Get dates in a single column
    dates = np.array( [data[:, meta.column_index['date']][i + history_points].copy() for i in range( len( data ) - history_points)] )
    #dates = np.array( [utils.convert_to_datetime( i ) for i in dates] ) # Convert back to normal dates

    technical_indicators_normalised = stock_indicators.get_technical_indicators( preprocessing.MinMaxScaler(), ohlcv_histories_normalised )

    assert ohlcv_histories_normalised.shape[0] == next_day_close_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    return dates, ohlcv_histories_normalised, technical_indicators_normalised, next_day_close_values_normalised, next_day_close_values, y_normaliser

class Preprocess:
    def __init__( self, test_split=False ):
        self.dates, self.ohlcv_histories, self.technical_indicators, \
        self.next_day_close_values, self.unscaled_y, \
        self.y_normaliser = prepare( get.dataset() )

        print( "\n\n** Print data shapes: " )
        print( "*********************************" )
        print( "dates:", len( self.dates ) )
        print( "ohlcv_histories:", len( self.ohlcv_histories ) )
        print( "technical_indicators:", len( self.technical_indicators ) )
        print( "next_day_close_values:", len( self.next_day_close_values ) )
        print( "unscaled_y:", len( self.unscaled_y ) )
        print( "*********************************\n\n" )

        if test_split:
            self.n_split = int( self.ohlcv_histories.shape[0] * test_split )

    def get_unscaled_data( self ):
        return ( self.unscaled_y[self.n_split:] )

    def get_training_data( self ):
        return ( self.ohlcv_histories[:self.n_split], self.technical_indicators[:self.n_split], self.next_day_close_values[:self.n_split], self.dates[:self.n_split] )

    def get_test_data( self ):
        return ( self.ohlcv_histories[self.n_split:], self.technical_indicators[self.n_split:], self.next_day_close_values[self.n_split:], self.dates[self.n_split:] )

    def get_y_normalizer( self ):
        return self.y_normaliser

    def get_history_for_date( self, date ):
        dates = np.array( [datetime.datetime.fromtimestamp( i ) for i in self.dates] )
        date_min = dates.min()
        date_max = dates.max()

        if ( date > date_min and date <= date_max ):
            idx = np.searchsorted( dates, date )
            return ( self.ohlcv_histories[idx], self.technical_indicators[idx], self.next_day_close_values[idx], self.dates[idx] )
        else:
            raise Exception( "Date ranges should be between '%s' & '%s'" % ( date_min.strftime( '%b %d, %Y' ), date_max.strftime( '%b %d, %Y' ) ) )
