import six
import json
import datetime
import pandas as pd

import ml_trader.utils.file as file
from ml_trader.config import Config as config

from pprint import pprint
from alpha_vantage.timeseries import TimeSeries


def modify_metadata( metadata ):
    updated_names_mapping = {
        "1. Information": "Information",
        "2. Symbol": "Symbol",
        "3. Last Refreshed": "Last Refreshed",
        "4. Output Size": "Output Size",
        "5. Time Zone": "Time Zone"
    }

    for oldname, newname in updated_names_mapping.items():
        metadata[newname] = metadata[oldname]
        del metadata[oldname]

    metadata['Last Downloaded'] = '{:%b %d, %Y %H:%M}'.format( datetime.datetime.now() )

    return metadata

def retrieve( symbol, time_window, force=False ):
    filepath = config.data_filepath % ( symbol.upper(), time_window )
    json_filepath = config.metadata_filepath.format( datetime.datetime.now() )

    if not file.exists( filepath ) or force != False:
        credentials = json.load( open( 'creds.json', 'r' ) )
        api_key = credentials['av_api_key']

        ts = TimeSeries( key=api_key, output_format='pandas' )

        print( "Downloading dataset..." )

        if time_window == 'intraday':
            data, meta_data = ts.get_intraday( symbol=symbol, interval='1min', outputsize='full' )

        elif time_window == 'daily':
            data, meta_data = ts.get_daily( symbol, outputsize='full' )

        elif time_window == 'daily_adj':
            data, meta_data = ts.get_daily_adjusted( symbol, outputsize='full' )

        # Create directory if don't exist
        file.create_path_if_needed( filepath )

        # Modify and save metadata
        with open( json_filepath, 'w', encoding='utf-8' ) as f:
            json.dump( modify_metadata( meta_data ), f )

        # Save to csv while respecting configuration
        data.to_csv( filepath )

        # Important to reset the index, so output is inline with csv import
        return data.reset_index()

    # If file already exist, just read and return data
    else:
        return pd.read_csv( filepath )

def dataset( *args, **kargs ):
    if len( kargs ) == 2 and 'symbol' in kargs and 'time_window' in kargs and \
        isinstance( kargs['symbol'], six.string_types ) and \
        isinstance( kargs['time_window'], six.string_types ):

        data = retrieve( kargs['symbol'], kargs['time_window'], force=True )

    elif len( args ) == 0:
        data = retrieve( config.stock_symbol, config.dataset_type )

    else:
        raise Exception( "Error!" )

    # Show dataframe stats
    print( "\nDataset for {} (first 50 rows):".format( config.stock_symbol.upper() ) )
    pprint( data.head( 50 ) )
    pprint( data.tail( 50 ) )
    print( "\n\n" )

    # Sort ascending and prevent date from being indexed
    return data
