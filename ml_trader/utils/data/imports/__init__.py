import ml_trader.utils.data.imports.get as get
import ml_trader.utils.data as dt


def from_csv():
    return dt.prepare( get.dataset() )
