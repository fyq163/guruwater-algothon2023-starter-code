import pandas as pd


def rest_import():
    print('import success')


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


def get_mom(df, time_frame):
    """
    :param df: dataframe of prices, cols are per stocks, rows are days
    :param time_frame: time frame for momentum
    :return: momentum for each stock
    In the first time_frame days, no data, no data, no actions.

    """
    (nins, nt) = df.shape


if __name__ == '__main__':
    pricesFile = "./prices.txt"
    prcAll = loadPrices(pricesFile)
    print("Loaded %d instruments for %d days" % (nInst, nt))
    prcHistSoFar = prcAll[:,:30]
    mom = get_mom(prcHistSoFar, 14)
