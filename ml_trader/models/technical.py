import numpy as np

import ml_trader.utils.file as file
import ml_trader.config as config
import ml_trader.utils.data.meta as meta

from ml_trader.models.importance_sampling.training import ImportanceTraining


class Technical_Model:
    model = False

    def __init__( self, y_normaliser ):
        # Used to invert normalision applied to predictions
        self.y_normaliser = y_normaliser
        self.data_column_size = len( meta.column_index )

    def build( self ):
        print( " **Initializing model..." )

        # Lazy loading...
        from keras import regularizers
        from keras import optimizers
        from keras.models import Model
        from keras.layers import Input
        from keras.layers import Dense
        from keras.layers import LSTM
        from keras.layers import Dropout
        from keras.layers import Activation
        from keras.layers import concatenate

        # define two sets of inputs
        lstm_input = Input( shape=( config.history_points, self.data_column_size ), name='lstm_input' )
        dense_input = Input( shape=( config.technical_indictors_input_size, ), name='tech_input' )

        # The first branch operates on the first input
        #x = LSTM( config.history_points, name='lstm_0' )( lstm_input )
        x = LSTM( 50, name='lstm_0' )( lstm_input )
        x = Dropout( 0.2, name='lstm_dropout_0')( x )
        #x = Dense( 64, input_dim=20, kernel_regularizer=regularizers.l2( 0.01 ) )( x )
        lstm_branch = Model( inputs=lstm_input, outputs=x )

        # the second branch opreates on the second input
        y = Dense( 20, name='tech_dense_0' )( dense_input )
        y = Activation( "relu", name='tech_relu_0' )( y )
        y = Dropout( 0.2, name='tech_dropout_0' )( y )
        #y = Dense( 64, input_dim=20, kernel_regularizer=regularizers.l2( 0.01 ) )( y )
        technical_indicators_branch = Model( inputs=dense_input, outputs=y )

        # combine the output of the two branches
        combined = concatenate( [lstm_branch.output, technical_indicators_branch.output], name='concatenate' )


        z = Dense( 64, activation="sigmoid", name='dense_pooling' )( combined )

        # Linear is better for regression instead of classification
        z = Dense( 1, activation="linear", name='dense_out' )( z )

        # This model will accept the inputs of the two branches and
        # then output a single value
        self.model = Model( inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z )
        adam = optimizers.Adam( lr=0.0005 )
        self.model.compile( optimizer=adam, loss='mse', metrics=['accuracy'] )

        return self._check_model( self.model )

    def _check_model( self, model ):
        print( "\n\n******************************************************" )
        print( model.summary() )
        print( "******************************************************\n\n" )

        if \
            model.layers[0].output_shape[1] != config.technical_indictors_input_size or \
            model.layers[1].output_shape[1] != config.history_points or \
            model.layers[1].output_shape[2] != self.data_column_size:
            print(
                '\n\nModel expects: ( %s, %s )\nData configured: ( %s, %s )\n'
                % ( model.layers[1].output_shape[1],
                model.layers[1].output_shape[2],
                config.history_points,
                self.data_column_size )
            )
            print( "*** Please retrain this model. Dataset or config is out of sync with the saved model!\n\n" )

            raise Exception( "\n\n!!! Model needs to be retrained!!!\n\n" )
        return self

    def _save_model_visualization( self, model ):
        file_path = file.timestamp( config.model_visualization_filepath )
        file.create_path_if_needed( file_path )

        from keras.utils import plot_model
        plot_model( model, to_file=file_path, show_shapes=True )

    def get_model( self ):
        return self.model

    def save( self ):
        file.create_path_if_needed( config.model_filepath )
        self.model.save( config.model_filepath )
        self._save_model_visualization( self.model )

    def score( self, x, y ):
        return self.model.evaluate( x, y )

    def mean_squaured_error( self, y_test, y_predicted ):
        '''
        Mean Squared Error Definition

        The mean squared error tells you how close a regression line is to a set of
        points. It does this by taking the distances from the points to the regression
        line (these distances are the “errors”) and squaring them. The squaring is
        necessary to remove any negative signs. It also gives more weight to larger
        differences. It’s called the mean squared error as you’re finding the average
        of a set of errors.
        '''
        real_mse = np.mean( np.square( y_test - y_predicted ) )
        return real_mse / ( np.max( y_test ) - np.min( y_test ) ) * 100

    def load( self ):
        if not file.exists( config.model_filepath ):
            raise Exception( "*** Model ('%s') does not exist! Please train." % config.model_filepath )

        # Lazy loading...
        from keras.models import load_model

        # Load pretrianed model
        self.model = load_model( config.model_filepath )

        # Check if model compatible with dataset
        return self._check_model( self.model )

    def optimized_training( self, x, y, x_test, y_test ):
        wrapped =  ImportanceTraining( self.model )
        wrapped.fit(
            x, y,
            batch_size=config.batch_size, epochs=config.epochs,
            validation_data=( x_test, y_test )
        )

    def train( self, x, y, x_test, y_test ):
        '''
        # x = new input data / features
        # y = predicted data

        # Sample - a single row of data
        # Batch - the number of samples to work through before updating the internal model parameters
        # Epoc - the number times that the learning algorithm will work through the entire training dataset

        Assume you have a dataset with 200 samples (rows of data) and you
        choose a batch size of 5 and 1,000 epochs.

        This means that the dataset will be divided into 40 batches, each with
        five samples. The model weights will be updated after each batch of
        five samples.

        This also means that one epoch will involve 40 batches or 40 updates
        to the model.
        '''

        #from keras.callbacks import EarlyStopping

        return self.model.fit( x=x, y=y, \
            batch_size=config.batch_size, epochs=config.epochs, \
            shuffle=config.shuffle, validation_split=0.1, \
            #callbacks=[EarlyStopping( monitor='val_loss', patience=10 )], \
            validation_data=(x_test, y_test), \
            verbose=1 )

    def predict( self, y ):
        y_predicted = self.model.predict( y )
        return self.y_normaliser.inverse_transform( y_predicted )

    def predict_raw( self, y ):
        return self.model.predict( y )
