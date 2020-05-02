import talib
from talib import MA_Type
import numpy as np
import pandas as pd
import math as m

from ml_trader.config import Config as config


def get_technical_indicators_talib( scaler, df, len ):
    tech_ind = pd.DataFrame()

    tech_ind["sma"] = talib.SMA( df.close, config.history_points )
    #tech_ind["macd"], _, _ = talib.MACD( ohlcv.close )
    #tech_ind["macd"], tech_ind["sigal"], tech_ind["hist"] = talib.MACD( df.close )
    #tech_ind["ma10"] = talib.MA ( df.close, timeperiod=10 )
    tech_ind["ma30"] = talib.MA( df.close, timeperiod=30 )
    #tech_ind["mom"] = talib.MOM( df.close, timeperiod=5 )
    #tech_ind["upper"], tech_ind["middle"], tech_ind["lower"] = talib.BBANDS( df.close, matype=MA_Type.T3 )
    #tech_ind['RSI'] = talib.RSI( df.close, timeperiod=14 )

    # Get last x rows ( exclude history_points )
    tech_ind = tech_ind.tail( len )

    # Auto set configuration
    config.technical_indictors_input_size = tech_ind.shape[1]
    tech_ind_normalized = scaler.fit_transform( np.array( tech_ind ) )
    return scaler, tech_ind_normalized
