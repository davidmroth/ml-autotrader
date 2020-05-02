import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import ml_trader.utils as utils
from ml_trader.config import Config as config
import ml_trader.utils.file as file


years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%Y')


def format_price( p ):
    return np.set_printoptions( formatter={'float': lambda x: "{0:0.2f}".format( price_today[0][0] )} )

class Plot:
    def __init__( self, name, legend=None, xlabel=None, ylabel=None, scalers=False  ):
        if scalers:
            self.y_scaler = scalers[config.label_column]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.name = name
        self.legend = legend

    def _unpack_buy_sells( self, value, index ):
        return list( list( zip( *value ) )[index] )

    def add_note( self, text ):
        textstr = '\n'.join( text )

        props = dict( boxstyle='round', facecolor='wheat', alpha=0.5 )
        plt.gcf().text(
            #0.135, 0.125,
            0.1, 0.1,
            textstr,
            fontsize=14,
            verticalalignment='top',
            bbox=props
        )

    def title( self, title ):
        plt.title( title )

    def graph( self, y_axis, x_axis=None, label=None, scale=False ):
        if scale:
            y_axis = self.y_scaler.inverse_transform( y_axis )

        if ( x_axis is None ):
            plt.plot( y_axis, label=label )
        else:
            x_axis = np.array( [datetime.datetime.fromtimestamp( i ) for i in x_axis] )

            ax = plt.gca()
            plt.plot( x_axis, y_axis, label=label )

            xmin, xmax = ax.get_xlim()
            custom_ticks = np.linspace( xmin, xmax, 10, dtype=int )
            ax.set_xticks( custom_ticks )
            ax.set_xticklabels( custom_ticks )
            ax.xaxis.set_major_formatter( mdates.DateFormatter( '"%B %d, %Y"' ) )

    def plot_buys_and_sells( self, x_axis, x_index, y_axis, y_index, c, s, label=None ):
        x = self._unpack_buy_sells( x_axis, x_index )
        x = np.array( [datetime.datetime.fromtimestamp( i ) for i in x] )
        y = self._unpack_buy_sells( y_axis, y_index )
        y = np.around( y, 2 )

        plt.scatter( x, y, c=c, s=s )

    def create( self ):
        file_path = file.timestamp( config.train_analysis.format( self.name + '_' + time.strftime( "%H_%M_%S" )) )
        file.create_path_if_needed( file_path )

        plt.gcf().set_size_inches( 22, 15, forward=True )

        if self.legend:
            plt.legend( self.legend )

        else:
            plt.legend()

        plt.xlabel( self.xlabel )
        plt.ylabel( self.ylabel )
        plt.savefig( file_path )
        plt.show()
        plt.clf()
