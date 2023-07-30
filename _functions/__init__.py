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

def get_mom(fn, time_frame, ):
    """

    :return:
    """

if __name__ == '__main__':
    pricesFile = "./prices.txt"
    prcAll = loadPrices(pricesFile)
    print("Loaded %d instruments for %d days" % (nInst, nt))