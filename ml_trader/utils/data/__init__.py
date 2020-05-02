import timeit
'''
start_time = timeit.default_timer()
some_function()
print( '{:.99f}'.format( timeit.default_timer() - start_time ).rstrip('0').rstrip('.') )
'''

import pandas as pd
import numpy as np
import datetime

import ml_trader.utils as utils
import ml_trader.utils.file as file
import ml_trader.utils.stock.indicators as stock_indicators
import ml_trader.utils.data.imports.get as get

from ml_trader.config import Config as config

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
    data_scalers = {}
    dm = utils.DateManager()
    history_points = config.history_points

    #
    # Feature engineering
    #

    # Modify column header names
    new_column_names = df.columns.values
    new_column_names[1] = 'open'
    new_column_names[2] = 'high'
    new_column_names[3] = 'low'
    new_column_names[4] = 'close'
    new_column_names[5] = 'volume'
    df.columns = new_column_names

    # Sort dataframe ascending & reset index
    df = df.sort_values( ['date'], ascending=True ).reset_index( drop=True )

    # Get next row's open value, and add it to current row as a new column
    df['next_day_change'] = df['open'].shift( -1 )

    # Change 'next_day_change' column to represent if the next day was: higher(1), lower(-1), or no change (0)
    df['next_day_change'] = np.where( df['next_day_change'] < df['open'], -1, df['next_day_change'] ).astype( int )
    df['next_day_change'] = np.where( df['next_day_change'] > df['open'], 1, df['next_day_change'] ).astype( int )
    df['next_day_change'] = np.where( df['next_day_change'] == df['open'], 0, df['next_day_change'] ).astype( int )

    df['date'] = pd.to_datetime( df['date'] ) # Convert to datetime
    df['weekday_num'] = df['date'].apply( lambda x: x.weekday() ) # Get weekday_num as a feature

    df['date'] = df['date'].apply( dm.convert_to_timestamp ) # Convert to unix timestamp which can be normalized

    #Label to predict
    df['label'] = df[config.label_column].shift( -config.look_ahead )

    # The first day of trading that stock often looked anomalous due to the massively high volume (IPO).
    # This inflated max volume value also affected how other volume values in the dataset were scaled when normalising the data,
    # so we drop the oldest data point out of every set)
    df = df.iloc[1:]

    #TODO: Check if correct
    # Lob off last x rows since we can't add future stock dates to the last x rows
    if ( config.look_ahead > 0 ):
        df = df.iloc[:-config.look_ahead]

    # Copy values to seperate numpy array
    labels = df['label'].values
    dates = df['date'].values
    next_day_change = df['next_day_change'].values

    # remove columns from data frame
    df = df.drop( 'date', axis=1 )
    df = df.drop( 'label', axis=1 )
    df = df.drop( 'next_day_change', axis=1 )
    #df = df.drop( 'weekday_num', axis=1 )

    df = df.reset_index()
    df = df.drop( 'index', axis=1 )

    #print( df )
    config.set( 'column_index', { key: value for (key,value) in zip( range(0, len( df.columns.values ) ), df.columns.values) } )

    data = df.values # Convert to numpy array
    #print( data )

    # Normalise the data — scale it between 0 and 1 — to improve how quickly
    # our network converges
    normaliser = preprocessing.MinMaxScaler()
    data_normalised = normaliser.fit_transform( data ) # Normalize all columns
    data_scalers['data'] = normaliser
    #print( data_normalised )

    # Num of data points
    num_data_points = len( data ) - history_points

    '''
    Using the last {history_points} open close high low volume data points, predict the next value
    Loop through all the stock data, and add build a normalized dataset that include x number of ohlcv history items for each stock date
    Lob off the first x items as they won't include x previous date
    x = history_points
    '''
    #TODO: Figure out why 'i+1:i + history_points+1' works, but not i:i + history_points
    #feat_ohlcv_histories_normalised = np.array( [data_normalised[i:i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    #feat_ohlcv_histories_normalised = np.array( [data_normalised[i+1:i + history_points].copy() for i in range( num_data_points )] )
    feat_ohlcv_histories_normalized = np.array( [data_normalised[i:i + history_points].copy() for i in range( num_data_points )] )
    #print( feat_ohlcv_histories_normalised )

    # Label data
    label_normalizer = preprocessing.MinMaxScaler()
    #NICE: (np.expand_dims) each item added to its own array and this is super fast
    labels_scaled = label_normalizer.fit_transform( np.expand_dims( labels[history_points:,], -1 ) )
    data_scalers[config.label_column] = label_normalizer
    #print( np.expand_dims( labels[history_points:,], -1 ) )
    #print( labels_scaled )
    #print( label_normalizer.inverse_transform( labels_scaled ) )

    # Get all labels minus last x days ( minus history_points )
    #labels_scaled = np.array( [labels_normalized[i + history_points].copy() for i in range( num_data_points )] )

    # Get all dates minus last x days ( minus history_points )
    dates = np.array( [dates[i + history_points].copy() for i in range( num_data_points )] )

    # Get normalize technical indictors
    data_scalers['tech_ind'], feat_technical_indicators_normalized = stock_indicators.get_technical_indicators_talib( preprocessing.MinMaxScaler(), df, num_data_points )

    print( feat_ohlcv_histories_normalized.shape[0], labels_scaled.shape[0], feat_technical_indicators_normalized.shape[0] )
    assert feat_ohlcv_histories_normalized.shape[0] == labels_scaled.shape[0] == feat_technical_indicators_normalized.shape[0]
    #print( '\n\nDates:', dates, '\n\nfeat_ohlcv_histories_normalised', feat_ohlcv_histories_normalised, '\n\nfeat_technical_indicators_normalised', feat_technical_indicators_normalised, '\n\nlabels_scaled', labels_scaled )

    return dates, feat_ohlcv_histories_normalized, feat_technical_indicators_normalized, labels_scaled, data_scalers

class Preprocess:
    def __init__( self, test_split=False ):
        data = get.dataset()

        self.dates, self.ohlcv_histories, self.technical_indicators, \
        self.scaled_y, self.data_scalers = prepare_labels_feat( data )

        print( "\n\n** Print data shapes: " )
        print( "*********************************" )
        print( "dates:", len( self.dates ) )
        print( "ohlcv_histories:", len( self.ohlcv_histories ) )
        print( "technical_indicators:", len( self.technical_indicators ) )
        print( "scaled_y:", len( self.scaled_y ) )
        print( "*********************************\n\n" )

        if test_split:
            self.n_split = int( self.ohlcv_histories.shape[0] * test_split )

    def get_unscaled_data( self ):
        return ( self.data_scalers[config.label_column].inverse_transform( self.scaled_y[self.n_split:] ) )

    def get_training_data( self ):
        return ( self.ohlcv_histories[:self.n_split], self.technical_indicators[:self.n_split], self.scaled_y[:self.n_split], self.dates[:self.n_split] )

    def get_test_data( self ):
        return ( self.ohlcv_histories[self.n_split:], self.technical_indicators[self.n_split:], self.scaled_y[self.n_split:], self.dates[self.n_split:] )

    def get_all_data( self ):
        return ( self.ohlcv_histories, self.technical_indicators, self.scaled_y, self.dates )

    def get_scalers( self ):
        return self.data_scalers

    def get_history_for_date( self, date ):
        dates = np.array( [datetime.datetime.fromtimestamp( i ) for i in self.dates] )
        date_min = dates.min()
        date_max = dates.max()

        if ( date > date_min and date <= date_max ):
            idx = np.searchsorted( dates, date )
            return ( self.ohlcv_histories[idx], self.technical_indicators[idx], self.scaled_y[idx], self.dates[idx] )
        else:
            raise Exception( "Date ranges should be between '%s' & '%s'" % ( date_min.strftime( '%b %d, %Y' ), date_max.strftime( '%b %d, %Y' ) ) )
