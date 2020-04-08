# Keep track of the number of {history_points} we want to use; the number of days
# of stock history the model gets to base its predictions off of. So,
# if history_points is set to 50, the model will train on and require the
# past 50 days of stock history to make a prediction about just the next day.
history_points = 50
num_data_columns = 5
model_file = 'models/technical_model.h5' #TODO: dynamic file name
data_file = 'data/MSFT_daily.csv' #TODO: dynamic stock symbol
prediction_analysis = 'analysis/predict_out_%s.png'
train_analysis = 'analysis/train_out_%s.png'
