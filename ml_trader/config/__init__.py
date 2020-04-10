# Keep track of the number of {history_points} we want to use; the number of days
# of stock history the model gets to base its predictions off of. So,
# if history_points is set to 50, the model will train on and require the
# past 50 days of stock history to make a prediction about just the next day.

stock_symbol = 'msft'
dataset_type = 'daily'

# File paths
data_filepath = 'data/stock/%s_%s.csv'
model_filepath = 'data/models/technical_model.h5'
prediction_analysis = 'data/analysis/{}_out_%s.png'
train_analysis = 'data/analysis/{}_out_%s.png'


# Trade Logic
purchase_amt = 10
trade_threshold = 0.1


# Model hyperparamters
history_points = 50 # Can't be less than 13
batch_size = 64
epochs = 50 # Number of training runs ( typlically, the more the better )
