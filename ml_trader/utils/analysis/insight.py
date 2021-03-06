import numpy as np

import ml_trader.utils as utils
import ml_trader.utils.data.meta as meta


def get_trade_insight( date, price_today, predicted_price_tomorrow ):
    # I can only predict the next day, but I need all of today's stats for today, so I
    # can only work at the end of the day. That means I have to buy stock
    # low or high on the following day based on the output of the model

    # Input the ochlv (for the previous 50 days?)
    # Output the following day's close price

    change = False
    delta = np.round( predicted_price_tomorrow - price_today, 2 ).item()
    day = utils.convert_to_datetime( date )

    #TODO: get actual next trading day
    #next_day = day + datetime.timedelta( days=1 )
    #TODO: Add trading after a holidy indictor for the model, so it will learn
    # to recognize trading patterns after a holiday (?)
    next_day = 'the next trading day'

    percent_change = abs( float( '{:.2f}'.format( ( 100 * ( 1 - ( predicted_price_tomorrow / price_today ) ) ).item() ) ) )
    decrease_increase = 'decrease' if delta < 0 else 'increase'
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

    print(
        '''
        Using data points for: {:%b %d, %Y}, I predict that the stock will {} on {} at
        '${:.2f}', {} over the stock
        price {} today at '${:.2f}'.
        '''.format(
            day,
            meta.label_column,
            next_day,
            predicted_price_tomorrow.item(),
            price_change_summary,
            meta.label_column,
            price_today[0][0].item()
        )
    )
