import pandas as pd
from scipy.ndimage import shift
import numpy as np


def rest_import():
    print('import success')


def test_get_mom(prc):
    """
    PASSED test
    """
    return get_mom(prc[0])


def loadPrices(fn):
    """
    :param fn: "./prices.txt"
    :return: a dataframe of prices, cols are per stocks, rows are days
    """
    global nt, nInst
    # df=pd.read_csv(fn, sep='\s+', names=cols, header=None, index_col=0)
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T


def get_mom(a_col, time_frame=14):
    """
    :param a_col: a single column ndarray of prices
    :param time_frame: time frame for momentum
    :return: momentum for this rol, this stock
    In the first time_frame days, no data, no data, no actions.
    """
    momentum_list = shift(a_col, -time_frame, cval=np.NaN) - a_col
    momentum_list = np.roll(momentum_list, np.count_nonzero(np.isnan(momentum_list)))
    return momentum_list


def get_macd(raw_prc, n_fast=12, n_slow=26):
    """
    Create a MACD indicator with the specified fast and slow moving
    :param raw_prc: take raw prcHistSoFar and process inside function
    :param n_fast: fast moving average default 12
    :param n_slow:default 26
    :return:macd, signal, histogram, we care only histogram and if it is changing sign.
    No format change, it is still column for date, rows for stocks
    """
    df = pd.DataFrame(raw_prc)
    ema_fast = df.ewm(span=n_fast, min_periods=n_slow, axis=1).mean()
    ema_slow = df.ewm(span=n_slow, min_periods=n_slow, axis=1).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, min_periods=8, axis=1).mean()
    histogram = macd - signal
    return macd, signal, histogram



if __name__ == '__main__':
    pricesFile = "../prices.txt"
    prcAll = loadPrices(pricesFile)
    print("Loaded %d instruments for %d days" % (nInst, nt))
    prcHistSoFar = prcAll[:, :50]
    _,_,histogram = get_macd(prcHistSoFar)
