import numpy as np

import ml_trader.utils as utils
from ml_trader.config import Config as config


class Insight:
    summary_data = []

    def __init__( self, data, dates ):
        # TODO: not completed; for getting next day's
        # data for better analysis
        self.dates = dates
        self.data = data

    def _change_text( self, delta ):
        if delta > 0:
            return 'increase'

        elif delta < 0:
            return 'decrease'
        else:
            return 'no change'

    def get_trade_insight( self, date, price_today, predicted_price_tomorrow, last=False, track_increase_decrease_predictions=False ):
        current_index =  np.where( self.dates == date )[0][0]
        price_today = price_today[0][0].item()
        predicted_price_tomorrow = predicted_price_tomorrow.item()

        # I can only predict the next day, but I need all of today's stats for today, so I
        # can only work at the end of the day. That means I have to buy stock
        # low or high on the following day based on the output of the model

        # Output the following day's close price
        change = False
        delta = np.round( predicted_price_tomorrow - price_today, 2 ).item()
        today = utils.convert_to_datetime( date )

        percent_change = abs( float( '{:.2f}'.format( 100 * ( 1 - ( predicted_price_tomorrow / price_today ) ) ) ) )
        #decrease_increase = 'decrease' if delta < 0 else 'increase'
        decrease_increase = self._change_text( delta )
        change_summary_text = 'which will represent a ${0:.2f} {1} of {2:.2f}%'

        if not ( delta > 0 or delta < 0 ):
            change_summary_text = 'which will represents no change'

        if percent_change < 1:
            change_summary_text = 'which will represent a ${0:.2f} {1} of less than 1%'

        price_change_summary = change_summary_text.format(
            abs( delta ),
            decrease_increase,
            percent_change
        )

        if last:
            next_business_day = "{:%b %d, %Y}".format(
                utils.DateManager()
                    .init_next_biz_day( today )
                    .get_next_trade_day( today )
            )
            track_increase_decrease_predictions.append( -1 )
        else:
            next_day = utils.convert_to_datetime( self.dates[current_index+1] )
            next_day_actual_price = self.data[current_index+1][0]
            actual = 'Actual: {0:%b %d, %Y} @ ${1:.2f} ({2})'.format(
                next_day,
                next_day_actual_price,
                self._change_text( next_day_actual_price - price_today )
            )
            #next_business_day = "the next trading day".format( next_day )
            next_business_day = "{:%b %d, %Y}".format( next_day )

            print(
                '''
                Using data points for: {:%b %d, %Y}, I predict that the stock will {} on {} at
                '${:.2f}', {} over the stock
                price {} today at '${:.2f}'.\n
                [{}]
                '''.format(
                    today,
                    config.label_column,
                    next_business_day,
                    predicted_price_tomorrow,
                    price_change_summary,
                    config.label_column,
                    price_today,
                    actual
                )
            )

            if not isinstance( track_increase_decrease_predictions, bool ):
                track_increase_decrease_predictions.append( decrease_increase == self._change_text( next_day_actual_price - price_today ) )
