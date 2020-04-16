import time
import json
import pytz
import datetime
import pandas as pd

import ml_trader.config as config


class DateManager:
    file_tz = pytz.utc
    local_tz = pytz.timezone( config.timezone )

    def __init__( self ):
        with open( config.metadata_filepath ) as f:
            data = json.load( f )

        self.file_tz = pytz.timezone( data['Time Zone'] )

    def convert_to_timestamp( self, x ):
        """Convert date objects to integers"""
        #NOTE: Timezone conversion is not needed
        #local_time = x.replace( tzinfo=self.file_tz ).astimezone( self.local_tz )
        #return time.mktime( local_time.to_pydatetime().timetuple() )
        return time.mktime( x.to_pydatetime().timetuple() )

def convert_to_datetime_str( x ):
    """Convert unix time (int) to date"""
    value = datetime.datetime.fromtimestamp( x )
    return f'{value:%Y-%m-%d}' #"%B %d, %Y"

def convert_to_datetime( x ):
    """Convert unix time (int) to date"""
    return datetime.datetime.fromtimestamp( x )
