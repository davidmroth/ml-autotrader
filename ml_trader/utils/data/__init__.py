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


def increase_decrease_same( df ):
    print( df.head() )
    if df['open_change'] > df['open']: return -1
    elif df['open_change'] < df['open']: return 1
    return 0

def prepare_labels_feat( df ):
    '''
    Transform data
    '''
    dm = utils.DateManager()
    history_points = config.history_points

    #
    # Feature engineering
    #

    df = df.sort_values( ['date'] ).reset_index( drop=True )

    # Modify column header names
    new_column_names = df.columns.values
    new_column_names[1] = 'open'
    new_column_names[2] = 'high'
    new_column_names[3] = 'low'
    new_column_names[4] = 'close'
    new_column_names[5] = 'volume'
    df.columns = new_column_names

    # Sort dataframe ascending
    df.sort_values( 'date', inplace=True, ascending=True )

    # Get next row's open value, and add it to current row as a new column
    df['open_change'] = df['open'].shift( -1 )

    # Change 'open_change' column to represent if the next day was: higher(1), lower(-1), or no change (0)
    df['open_change'] = np.where( df['open_change'] < df['open'], -1, df['open_change'] ).astype( int )
    df['open_change'] = np.where( df['open_change'] > df['open'], 1, df['open_change'] ).astype( int )
    df['open_change'] = np.where( df['open_change'] == df['open'], 0, df['open_change'] ).astype( int )

    df['date'] = pd.to_datetime( df['date'] ) # Convert to datetime
    df['weekday_num'] = df['date'].apply( lambda x: x.weekday() ) # Get weekday_num as a feature
    df['date'] = df['date'].apply( dm.convert_to_timestamp ) # Convert to unix timestamp which can be normalized

    # Copy values to seperate numpy array
    dates = df['date'].values
    open_change = df['open_change'].values

    # remove columns from data frame
    df = df.drop( 'date', axis=1 )
    df = df.drop( 'open_change', axis=1 )

    # The first day of trading that stock often looked anomalous due to the massively high volume (IPO).
    # This inflated max volume value also affected how other volume values in the dataset were scaled when normalising the data,
    # so we drop the oldest data point out of every set)
    data = df.iloc[1:].values # Convert to numpy array

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
    feat_ohlcv_histories_normalised = np.array( [data_normalised[i+1:i + history_points+1].copy() for i in range( len( data_normalised ) - history_points )] )
    #feat_ohlcv_histories_normalised = np.array( [data_normalised[i:i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    #feat_ohlcv_histories = np.array( [data[i:i + history_points].copy() for i in range( len( data ) - history_points )] )

    # Get normalized 'close' values, so model can be trained to predict this item
    labels_scaled = np.array( [data_normalised[:, meta.column_index[meta.label_column]][i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    labels_scaled = np.expand_dims( labels_scaled, -1 ) #NICE: each item added to its own array and this is super fast

    labels_unscaled = np.array( [data[:, meta.column_index[meta.label_column]][i + history_points].copy() for i in range( len( data ) - history_points )] )
    labels_unscaled = np.expand_dims( labels_unscaled, -1 ) #NICE: each item added to its own array and this is super fast

    label_normaliser = preprocessing.MinMaxScaler()
    label_normaliser.fit( labels_unscaled )

    # Normalize technical indictors
    feat_technical_indicators_normalised = stock_indicators.get_technical_indicators( preprocessing.MinMaxScaler(), feat_ohlcv_histories_normalised )

    # Get dates in a single column
    dates = np.array( [dates[i+1+history_points].copy() for i in range( len( data ) - history_points )] )

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
        print( self.dates )
        dates = np.array( [datetime.datetime.fromtimestamp( i ) for i in self.dates] )
        date_min = dates.min()
        date_max = dates.max()

        if ( date > date_min and date <= date_max ):
            idx = np.searchsorted( dates, date )
            return ( self.ohlcv_histories[idx], self.technical_indicators[idx], self.scaled_y[idx], self.dates[idx] )
        else:
            raise Exception( "Date ranges should be between '%s' & '%s'" % ( date_min.strftime( '%b %d, %Y' ), date_max.strftime( '%b %d, %Y' ) ) )
