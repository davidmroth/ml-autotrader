import os, errno
import time


def timestamp( filename ):
    datetime_str = time.strftime( "%Y_%m_%d" )
    return filename % datetime_str

def get_path( filename ):
    return os.path.dirname( filename )

def get_filename( filename ):
    return os.path.dirname( filename )

def get_filename_without_extension( filename ):
    file_basename = os.path.basename( filename )
    filename_without_extension = file_basename.split( '.' )[0]
    return filename_without_extension

def create_path_if_needed( filename ):
    directory = os.path.dirname( filename )

    try:
        os.makedirs( directory )

    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def exists( path ):
    if os.path.isfile( path ) and os.access( path, os.R_OK):
        # print( "File exists and is readable" )
        return True
    else:
        # print( "Either the file is missing or not readable" )
        return False
