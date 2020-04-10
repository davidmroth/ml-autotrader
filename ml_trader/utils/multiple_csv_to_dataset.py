import os
import numpy as np


def multiple_csv_to_dataset( test_set_name ):
    ohlcv_histories = 0
    technical_indicators = 0
    next_day_open_values = 0

    for csv_file_path in list( filter( lambda x: x.endswith( 'daily.csv' ), os.listdir( './' ) ) ):
        if not csv_file_path == test_set_name:
            print( csv_file_path )

            if type( ohlcv_histories ) == int:
                ohlcv_histories, technical_indicators, next_day_open_values, _, _ = csv_to_dataset( csv_file_path )

            else:
                a, b, c, _, _ = csv_to_dataset( csv_file_path )
                ohlcv_histories = np.concatenate( ( ohlcv_histories, a ), 0 )
                technical_indicators = np.concatenate( ( technical_indicators, b ), 0 )
                next_day_open_values = np.concatenate( ( next_day_open_values, c ), 0 )

    ohlcv_train = ohlcv_histories
    tech_ind_train = technical_indicators
    y_train = next_day_open_values

    ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser = csv_to_dataset( test_set_name )

    return ohlcv_train, tech_ind_train, y_train, ohlcv_test, tech_ind_test, y_test, unscaled_y_test, y_normaliser
