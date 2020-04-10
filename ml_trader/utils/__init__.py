import time
import datetime
import pandas as pd


def convert_to_timestamp( x ):
    """Convert date objects to integers"""
    return time.mktime( x.to_pydatetime().timetuple() )


def convert_to_datetime( x ):
    """Convert unix time (int) to date"""
    value = datetime.datetime.fromtimestamp( x )
    return f'{value:%Y-%m-%d}' #"%B %d, %Y"
