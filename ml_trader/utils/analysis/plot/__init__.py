import matplotlib.pyplot as plt

import ml_trader.config as config
import ml_trader.utils.file as file


class Plot:
    def __init__( self, name, start, end, legend  ):
        self.name = name
        self.legend = legend
        self.start = start
        self.end = end

    def _unpack_buy_sells( self, value, index ):
        return list( list( zip( *value ) )[index] )

    def add_note( self, text ):
        textstr = '\n'.join( text )

        props = dict( boxstyle='round', facecolor='wheat', alpha=0.5 )
        plt.gcf().text(
            0.135, 0.125,
            textstr,
            fontsize=14,
            verticalalignment='bottom',
            bbox=props
        )

    def graph( self, x_axis, y_axis, label=None ):
        plt.plot( x_axis[self.start:self.end], y_axis[self.start:self.end], label=label )

    def plot_buys_and_sells( self, x_axis, x_index, y_axis, y_index, c, s, label=None ):
        plt.scatter(
            self._unpack_buy_sells( x_axis, x_index ),
            self._unpack_buy_sells( y_axis, y_index ),
            c=c, s=s
        )

    def create( self ):
        file_path = file.timestamp( config.train_analysis.format( self.name ) )
        file.create_path_if_needed( file_path )

        plt.gcf().set_size_inches( 22, 15, forward=True )
        plt.legend( self.legend )
        plt.savefig( file_path )
        plt.show()