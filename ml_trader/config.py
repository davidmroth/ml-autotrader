class Config:
    __defaults = {
        # Stock settings
        'stock_symbol': 'spy',
        'dataset_type': 'daily',

        # User preferences
        'timezone': 'America/Chicago',


        # File paths
        'data_filepath': 'data/stock/%s_%s.csv',
        'metadata_filepath': 'data/stock/meta_data.json',
        'model_filepath': 'data/models/technical_model.h5',
        'model_visualization_filepath': 'data/analysis/model_visualization/%s.png',
        'prediction_analysis': 'data/analysis/pridictions/%s/{}.png',
        'train_analysis': 'data/analysis/training/%s/{}.png',


        # Trade Logic
        'purchase_amt': 10,
        'trade_threshold': 0.1,


        # Model hyperparamters
        # Keep track of the number of {history_points} we want to use; the number of days
        # of stock history the model gets to base its predictions off of. So,
        # if history_points is set to 50, the model will train on and require the
        # past 50 days of stock history to make a prediction about just the next day.

        'label_column': 'close',
        'look_ahead': 0,
        'history_points': 50, # Size of LSTM input / Can't be less than 13
        #technical_indictors_input_size: 10 # Size of technical indictors input / set automatically
        'shuffle': True,
        'batch_size': 32,
        'epochs': 300, # Number of training runs ( typlically, the more the better )
        'patience': 100,
    }

    __meta = {
        'column_index': {}
    }

    __setters = ["column_index"]

    @staticmethod
    def init():
        for name, value in Config.__defaults.items():
            setattr( Config, name, value )

        return Config

    @staticmethod
    def config( name ):
        return getattr( Config, name )

    @staticmethod
    def set( name, value ):
        if name in Config.__setters:
            Config.__meta[name] = value
            setattr( Config, name, value )

        else:
            raise NameError( "Name not accepted in set() method" )

config = Config.init()
