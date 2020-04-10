import ml_trader.config as config

from ml_trader.utils.imports.dataset import from_csv


class Preprocess:
    def __init__( self, test_split ):
        self.date_time, self.ohlcv_histories, self.technical_indicators, self.next_day_close_values, \
        self.unscaled_y, self.y_normaliser = from_csv()

        self.n_split = int( self.ohlcv_histories.shape[0] * test_split )

    def get_unscaled_data( self ):
        return ( self.unscaled_y[self.n_split:], self.date_time[self.n_split:] )

    def get_training_data( self ):
        return ( self.ohlcv_histories[:self.n_split], self.technical_indicators[:self.n_split], self.next_day_close_values[:self.n_split] )

    def get_test_data( self ):
        return ( self.ohlcv_histories[self.n_split:], self.technical_indicators[self.n_split:], self.next_day_close_values[self.n_split:] )

    def get_y_normalizer( self ):
        return self.y_normaliser
