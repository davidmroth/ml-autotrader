import time
import pytz
import datetime
import pandas as pd

import ml_trader.config as config


local_tz = pytz.timezone( config.timezone )

def convert_to_timestamp( x ):
    """Convert date objects to integers"""
    local_time = x.replace( tzinfo=pytz.utc ).astimezone( local_tz )
    return time.mktime( local_time.to_pydatetime().timetuple() )

def convert_to_datetime_str( x ):
    """Convert unix time (int) to date"""
    value = datetime.datetime.fromtimestamp( x )
    return f'{value:%Y-%m-%d}' #"%B %d, %Y"

def convert_to_datetime( x ):
    """Convert unix time (int) to date"""
    return datetime.datetime.fromtimestamp( x )
