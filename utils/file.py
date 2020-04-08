import time


def timestamp_file( filename ):
    datetime_str = time.strftime( "%m_%d_%Y-%H_%M_%S" )
    return filename % datetime_str
