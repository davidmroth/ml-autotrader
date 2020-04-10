import time
import datetime


def convert_to_timestamp( x ):
    """Convert date objects to integers"""
    return time.mktime( x.to_pydatetime().timetuple() )

def convert_to_datetime( x ):
    value = datetime.datetime.fromtimestamp( x )
    return f'{value:%Y-%m-%d}'
