'''
import timeit
start_time = timeit.default_timer()
some_function()
print( '{:.99f}'.format( timeit.default_timer() - start_time ))
'''

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
    # DEBUG:
    np.set_printoptions( suppress=True )
    pd.set_option( 'display.float_format', '{:f}'.format )

    data.sort_values( 'date', inplace=True, ascending=True )
    data['date'] = pd.to_datetime( data['date'] ) # Convert to datetime
    data['date'] = data['date'].apply( utils.convert_to_timestamp ) # Convert to unix timestamp which can be normalized

    data = data.tail( -1 )
    data = data.values

    # Normalise the data — scale it between 0 and 1 — to improve how quickly our network converges
    normaliser = preprocessing.MinMaxScaler()
    data_normalised = normaliser.fit_transform( data ) # Normalize all columns


    '''
    Using the last {history_points} open close high low volume data points, predict the next value
    Loop through all the stock data, and add build a normalized dataset that include x number of ohlcv history items for each stock date
    Lob off the first x items as they won't include x previous date
    x = history_points
    '''
    ohlcv_histories_normalised = np.array( [data_normalised[i:i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )


    # DEBUG
    ohlcv_histories_close = np.array( [data[:, meta.column_index['close']][i + history_points].copy() for i in range( len( data ) - history_points )] )
    ohlcv_histories_close = np.expand_dims( ohlcv_histories_close, -1 )

    print( "ohlcv_histories_close:", ohlcv_histories_close )

    close_normaliser = preprocessing.MinMaxScaler()
    ohlcv_histories_close_normalized = close_normaliser.fit_transform( ohlcv_histories_close )
    close_normaliser.fit( ohlcv_histories_close )

    print( 'First (scaled):', ohlcv_histories_close_normalized )
    print( ohlcv_histories_normalised[:,4] )
    print( ohlcv_histories_normalised[-1][4] )
    print( 'First:', close_normaliser.inverse_transform( np.array( [[ohlcv_histories_normalised[-1][0][0]]] ) ) )
    print( 'Last:', close_normaliser.inverse_transform( np.array( [[ohlcv_histories_normalised[-1][-1][4]]] ) ) )
    #exit()
    #End DEBUG

    # Get normalized 'close' values, so model can be trained to predict this item
    scaled_y = np.array( [data_normalised[:, meta.column_index['close']][i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    scaled_y = np.expand_dims( scaled_y, -1 ) #NICE: each item added to its own array and this is super fast

    unscaled_y = np.array( [data[:, meta.column_index['close']][i + history_points].copy() for i in range( len( data ) - history_points )] )
    unscaled_y = np.expand_dims( unscaled_y, -1 ) #NICE: each item added to its own array and this is super fast

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit( unscaled_y )

    # Get dates in a single column
    dates = np.array( [data[:, meta.column_index['date']][i + history_points].copy() for i in range( len( data ) - history_points)] )

    # Normalize technical indictors
    technical_indicators_normalised = stock_indicators.get_technical_indicators( preprocessing.MinMaxScaler(), ohlcv_histories_normalised )

    assert ohlcv_histories_normalised.shape[0] == scaled_y.shape[0] == technical_indicators_normalised.shape[0]
    return dates, ohlcv_histories_normalised, technical_indicators_normalised, scaled_y, unscaled_y, y_normaliser

class Preprocess:
    def __init__( self, test_split=False ):
        self.dates, self.ohlcv_histories, self.technical_indicators, \
        self.scaled_y, self.unscaled_y, \
        self.y_normaliser = prepare( get.dataset() )

        print( "\n\n** Print data shapes: " )
        print( "*********************************" )
        print( "dates:", len( self.dates ) )
        print( "ohlcv_histories:", len( self.ohlcv_histories ) )
        print( "technical_indicators:", len( self.technical_indicators ) )
        print( "scaled_y:", len( self.scaled_y ) )
        print( "unscaled_y:", len( self.unscaled_y ) )
        print( "*********************************\n\n" )

        if test_split:
            self.n_split = int( self.ohlcv_histories.shape[0] * test_split )

    def get_unscaled_data( self ):
        return ( self.unscaled_y[self.n_split:] )

    def get_training_data( self ):
        return ( self.ohlcv_histories[:self.n_split], self.technical_indicators[:self.n_split], self.scaled_y[:self.n_split], self.dates[:self.n_split] )

    def get_test_data( self ):
        return ( self.ohlcv_histories[self.n_split:], self.technical_indicators[self.n_split:], self.scaled_y[self.n_split:], self.dates[self.n_split:] )

    def get_y_normalizer( self ):
        return self.y_normaliser

    def get_history_for_date( self, date ):
        dates = np.array( [datetime.datetime.fromtimestamp( i ) for i in self.dates] )
        date_min = dates.min()
        date_max = dates.max()

        if ( date > date_min and date <= date_max ):
            idx = np.searchsorted( dates, date )
            return ( self.ohlcv_histories[idx], self.technical_indicators[idx], self.scaled_y[idx], self.dates[idx] )
        else:
            raise Exception( "Date ranges should be between '%s' & '%s'" % ( date_min.strftime( '%b %d, %Y' ), date_max.strftime( '%b %d, %Y' ) ) )
