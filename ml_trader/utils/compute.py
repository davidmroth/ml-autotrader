import ml_trader.config as config


def earnings( buys_, sells_ ):
    purchase_amt = config.purchase_amt
    stock = 0
    balance = 0

    while len( buys_ ) > 0 and len( sells_ ) > 0:
        if buys_[0][0] < sells_[0][0]:
            # time to buy $10 worth of stock
            balance -= purchase_amt
            stock += purchase_amt / buys_[0][1]
            buys_.pop( 0 )

        else:
            # time to sell all of our stock
            balance += stock * sells_[0][1]
            stock = 0
            sells_.pop( 0 )

    print( "Earnings: $%.2f" % balance )
