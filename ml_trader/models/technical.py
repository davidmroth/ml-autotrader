import ml_trader.utils.lazy as lazy_import
import ml_trader.utils.file as file
import ml_trader.config as config
import ml_trader.utils.data.meta as meta

# Lazy loading...
optimizers = lazy_import.lazy_module( 'keras.optimizers' )
Model = lazy_import.lazy_callable( 'keras.models.Model' )
Dense = lazy_import.lazy_callable( 'keras.layers.Dense' )
Input = lazy_import.lazy_callable( 'keras.layers.Input' )
LSTM = lazy_import.lazy_callable( 'keras.layers.LSTM' )
Dropout = lazy_import.lazy_callable( 'keras.layers.Dropout' )
Activation = lazy_import.lazy_callable( 'keras.layers.Activation' )
concatenate = lazy_import.lazy_callable( 'keras.layers.concatenate' )
load_model = lazy_import.lazy_callable( 'keras.models.load_model' )

from keras.callbacks import EarlyStopping


class Technical_Model:
    model = False

    def __init__( self, y_normaliser ):
        # Used to invert normalision applied to predictions
        self.y_normaliser = y_normaliser

    def build( self, input_size ):
        print( " **Initializing model..." )

        # define two sets of inputs
        lstm_input = Input( shape=( config.history_points, meta.column_count ), name='lstm_input' )
        dense_input = Input( shape=( input_size, ), name='tech_input' )

        # the first branch operates on the first input
        #x = LSTM( config.history_points, name='lstm_0' )( lstm_input )
        x = LSTM( 50, name='lstm_0' )( lstm_input )
        x = Dropout( 0.2, name='lstm_dropout_0')( x )
        lstm_branch = Model( inputs=lstm_input, outputs=x )

        # the second branch opreates on the second input
        y = Dense( 20, name='tech_dense_0' )( dense_input )
        y = Activation( "relu", name='tech_relu_0' )( y )
        y = Dropout( 0.2, name='tech_dropout_0' )( y )
        technical_indicators_branch = Model( inputs=dense_input, outputs=y )

        # combine the output of the two branches
        combined = concatenate( [lstm_branch.output, technical_indicators_branch.output], name='concatenate' )

        z = Dense( 64, activation="sigmoid", name='dense_pooling' )( combined )
        z = Dense( 1, activation="linear", name='dense_out' )( z )

        # This model will accept the inputs of the two branches and
        # then output a single value
        self.model = Model( inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z )
        adam = optimizers.Adam( lr=0.0005 )
        self.model.compile( optimizer=adam, loss='mse' )

        return self._validate_model( self.model )

    def _validate_model( self, model ):
        print( "\n\n******************************************************" )
        print( model.summary() )
        print( "******************************************************\n\n" )

        if model.layers[1].output_shape[1] != config.history_points or model.layers[1].output_shape[2] != meta.column_count:
            raise Exception( "*** Please retrain this model. Config is out of sync with the saved model!" )

        return model

    def get_model( self ):
        return self.model

    def score( self, x, y ):
        return self.model.evaluate( x, y, batch_size=config.batch_size )

    def load( self ):
        if not file.exists( config.model_filepath ):
            raise Exception( "*** Model ('%s') does not exist! Please train." % config.model_filepath )

        self.model = load_model( config.model_filepath )
        return self._validate_model( self.model )

    def save( self ):
        file.create_path_if_needed( config.model_filepath )
        self.model.save( config.model_filepath )

    def train( self, x, y, x_test, y_test ):
        # x = new input data
        # y = predicted data
        return self.model.fit( x=x, y=y, \
            batch_size=config.batch_size, epochs=config.epochs, \
            shuffle=False, validation_split=0.1, \
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)], \
            validation_data=(x_test, y_test), \
            verbose=1 )

    def predict( self, y ):
        y_predicted = self.model.predict( y )
        return self.y_normaliser.inverse_transform( y_predicted )
