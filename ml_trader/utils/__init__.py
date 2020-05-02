import time
import json
import pytz
import datetime
import pandas as pd

from datetime import datetime, timedelta, date
from dateutil import rrule

from ml_trader.config import Config as config


class DateManager:
    file_tz = pytz.utc
    local_tz = pytz.timezone( config.timezone )

    def __init__( self ):
        with open( config.metadata_filepath ) as f:
            data = json.load( f )

        self.file_tz = pytz.timezone( data['Time Zone'] )

    def init_next_biz_day( self, startday ):
        self.startday = startday

        holidays = [
            '5/25/2020',
            '7/3/2020',
            '9/7/2020',
            '11/26/2020',
            '12/25/2020'
        ]

        # Create a rule to recur every weekday starting today
        r = rrule.rrule(
            rrule.DAILY,
            byweekday = [rrule.MO, rrule.TU, rrule.WE, rrule.TH, rrule.FR],
            dtstart = startday
        )

        # Create a rruleset
        rs = rrule.rruleset()

        # Attach our rrule to it
        rs.rrule( r )

        # Add holidays as exclusion days
        for exdate in holidays:
            rs.exdate( datetime.strptime( exdate, '%m/%d/%Y' ) )

        self.next_biz_day_index = rs

        return self

    def convert_to_timestamp( self, x ):
        """Convert date objects to integers"""
        #NOTE: Timezone conversion is not needed
        #local_time = x.replace( tzinfo=self.file_tz ).astimezone( self.local_tz )
        #return time.mktime( local_time.to_pydatetime().timetuple() )
        return time.mktime( x.to_pydatetime().timetuple() )

    def get_next_trade_day( self, day ):
        index = ( self.startday - day ).days + 1
        return self.next_biz_day_index[index]

def convert_to_datetime_str( x ):
    """Convert unix time (int) to date"""
    value = datetime.fromtimestamp( x )
    return f'{value:%Y-%m-%d}' #"%B %d, %Y"

def convert_to_datetime( x ):
    """Convert unix time (int) to date"""
    return datetime.fromtimestamp( x )
