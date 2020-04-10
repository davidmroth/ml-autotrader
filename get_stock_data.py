import argparse
import ml_trader.utils.imports.dataset.get as get

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( 'symbol', type=str, help="the stock symbol you want to download" )
    parser.add_argument( 'time_window', type=str, choices=['intraday', 'daily', 'daily_adj'], help="the time period you want to download the stock history for" )

    namespace = parser.parse_args()
    print( namespace )
    get.dataset( **vars( namespace ) )
