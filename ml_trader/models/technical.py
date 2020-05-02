import os
import datetime
import numpy as np

import ml_trader.utils.file as file

from ml_trader.config import Config as config
from ml_trader.models.importance_sampling.training import ImportanceTraining


class Technical_Model:
    model = False
    y_predicted = False

    def __init__( self, data_scalers ):
        self.model_name = 'my_model'

        # Used to invert normalision applied to predictions
        self.data_scalers = data_scalers
        self.data_column_size = len( config.column_index )

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
        self.model.compile( optimizer=adam, loss='mse', metrics=['accuracy', 'mean_absolute_error'] )

        return self._check_model( self.model )

    def _check_model( self, model ):
        print( "\n\n******************************************************" )
        print( model.summary() )
        print( "******************************************************\n\n" )

        print(
            '\n\nModel expects: ( %s, %s, %s )\nFeats configured: ( %s, %s, %s )\n'
            % (
                model.layers[1].output_shape[2],
                model.layers[1].output_shape[1],
                model.layers[0].output_shape[1],
                self.data_column_size,
                config.history_points,
                config.technical_indictors_input_size
            )
        )

        if \
            model.layers[0].output_shape[1] != config.technical_indictors_input_size or \
            model.layers[1].output_shape[1] != config.history_points or \
            model.layers[1].output_shape[2] != self.data_column_size:
            print( "*** Please retrain this model. Dataset or config is out of sync with the saved model!\n\n" )

            raise Exception( "\n\n!!! Model needs to be retrained!!!\n\n" )

        # Save a visualization of the current model
        self._save_model_visualization( self.model )
        return self

    def _save_model_visualization( self, model ):
        file_path = file.timestamp( config.model_visualization_filepath )
        file.create_path_if_needed( file_path )

        from keras.utils import plot_model
        plot_model( model, to_file=file_path, show_shapes=True )

    def get_model( self ):
        return self.model

    def score( self, x, y ):
        return self.model.evaluate( x, y )

    def mean_squaured_error( self, y_test, y_hat_test ):
        '''
        What does the Mean Squared Error Tell You?

        Mean Absolute Error (MAE) is one of the most common metrics used to measure
        accuracy for continuous variables

        MAE measures the average magnitude of the errors in a set of predictions,
        without considering their direction.

        The smaller the means squared error, the closer you are to finding the line of
        best fit. Depending on your data, it may be impossible to get a very small
        value for the mean squared error. For example, the above data is scattered
        wildly around the regression line, so 6.08 is as good as it gets (and is in
        fact, the line of best fit).
        '''
        unscaled_y_test = self.data_scalers[config.label_column].inverse_transform( y_test )
        real_mse = np.mean( np.square( unscaled_y_test - y_hat_test ) )
        return real_mse / ( np.max( unscaled_y_test ) - np.min( unscaled_y_test ) ) * 100

    def root_mean_squared_error( self, y_test, y_hat_test ):
        '''
        What does the Root Mean Squared Error Tell You?

        Root mean squared error (RMSE) is a quadratic scoring rule that also
        measures the average magnitude of the error.

        Taking the square root of the average squared errors has some interesting
        implications for RMSE. Since the errors are squared before they are
        averaged, the RMSE gives a relatively high weight to large errors.
        This means the RMSE should be more useful when large errors are
        particularly undesirable.

        Conclusion
        RMSE has the benefit of penalizing large errors more, so it can be more
        appropriate in some cases, for example, if being off by 10 is more than
        twice as bad as being off by 5. But if being off by 10 is just twice as
        bad as being off by 5, then MAE is more appropriate.
        '''
        unscaled_y_test = self.data_scalers[config.label_column].inverse_transform( y_test )
        return np.sqrt( ( ( y_hat_test - unscaled_y_test ) ** 2 ).mean() )

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
        ImportanceTraining( self.model ).fit(
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

        from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

        # Get path and filename from config
        file.create_path_if_needed( config.model_filepath )
        dir = os.path.dirname( config.model_filepath )
        fname = os.path.basename( config.model_filepath )

        checkpointer = ModelCheckpoint( os.path.join( dir, fname ), save_best_only=True, verbose=1 )
        log_dir = "/tmp/tensorboard/" + datetime.datetime.now().strftime( "%Y%m%d-%H%M%S" )
        tensorboard = TensorBoard( log_dir=os.path.join( log_dir, self.model_name ) )

        return self.model.fit( x=x, y=y, \
            batch_size=config.batch_size, epochs=config.epochs, \
            shuffle=config.shuffle, validation_split=0.1, \
            callbacks=[EarlyStopping( monitor='val_loss', patience=config.patience ), tensorboard, checkpointer], \
            #callbacks=[tensorboard, checkpointer], \
            validation_data=(x_test, y_test), \
            verbose=1 )

    def predict( self, y ):
        y_hat = self.model.predict( y )
        return self.data_scalers[config.label_column].inverse_transform( y_hat )

    def predict_raw( self, y ):
        return self.model.predict( y )
