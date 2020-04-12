import timeit
'''
start_time = timeit.default_timer()
some_function()
print( '{:.99f}'.format( timeit.default_timer() - start_time ).rstrip('0').rstrip('.') )
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

#from ml_trader.utils.compute import earnings


def prepare_labels_feat( data ):
    '''
    Transform data
    '''

    history_points = config.history_points

    # The first day of trading that stock often looked anomalous due to the massively high volume (IPO).
    # This inflated max volume value also affected how other volume values in the dataset were scaled when normalising the data,
    # so we drop the oldest data point out of every set)
    data.sort_values( 'date', inplace=True, ascending=True )
    data['date'] = pd.to_datetime( data['date'] ) # Convert to datetime
    data['weekday_num'] = data['date'].apply( lambda x: x.weekday() ) # Get weekday_num as a feature
    data['date'] = data['date'].apply( utils.convert_to_timestamp ) # Convert to unix timestamp which can be normalized
    dates = data['date'].values

    # remove date column
    data = data.drop( 'date', axis=1 )

    # Remove first date since IPO's tend to swing wildly on the first day
    # of open and may confuse the model
    data = data.iloc[1:].values # Convert to numpy array

    # Normalise the data — scale it between 0 and 1 — to improve how quickly our network converges
    normaliser = preprocessing.MinMaxScaler()
    data_normalised = normaliser.fit_transform( data ) # Normalize all columns


    '''
    Using the last {history_points} open close high low volume data points, predict the next value
    Loop through all the stock data, and add build a normalized dataset that include x number of ohlcv history items for each stock date
    Lob off the first x items as they won't include x previous date
    x = history_points
    '''
    #TODO: Figure out why 'i+1:i + history_points+1' works, but not i:i + history_points
    #feat_ohlcv_histories_normalised = np.array( [data_normalised[i+1:i + history_points+1].copy() for i in range( len( data_normalised ) - history_points )] )
    feat_ohlcv_histories_normalised = np.array( [data_normalised[i:i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )

    # Normalize technical indictors
    feat_technical_indicators_normalised = stock_indicators.get_technical_indicators( preprocessing.MinMaxScaler(), feat_ohlcv_histories_normalised )

    # Get normalized 'close' values, so model can be trained to predict this item
    labels_scaled = np.array( [data_normalised[:, meta.column_index[meta.label_column]][i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    labels_scaled = np.expand_dims( labels_scaled, -1 ) #NICE: each item added to its own array and this is super fast

    labels_unscaled = np.array( [data[:, meta.column_index[meta.label_column]][i + history_points].copy() for i in range( len( data ) - history_points )] )
    labels_unscaled = np.expand_dims( labels_unscaled, -1 ) #NICE: each item added to its own array and this is super fast

    label_normaliser = preprocessing.MinMaxScaler()
    label_normaliser.fit( labels_unscaled )

    # Get dates in a single column
    dates = np.array( [dates[i + history_points].copy() for i in range( len( data ) - history_points )] )

    assert feat_ohlcv_histories_normalised.shape[0] == labels_scaled.shape[0] == feat_technical_indicators_normalised.shape[0]
    return dates, feat_ohlcv_histories_normalised, feat_technical_indicators_normalised, labels_scaled, labels_unscaled, label_normaliser

class Preprocess:
    def __init__( self, test_split=False ):
        self.dates, self.ohlcv_histories, self.technical_indicators, \
        self.scaled_y, self.unscaled_y, \
        self.y_normaliser = prepare_labels_feat( get.dataset() )

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
