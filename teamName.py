#!/usr/bin/env python
import _functions as f
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

nInst = 50
currentPos = np.zeros(nInst)
num_states = 4

model = hmm.GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, init_params='')


def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if (nt < 10):  # 前十天不做交易
        return np.zeros(nins)

    returns = np.diff(prcSoFar, axis=1) / prcSoFar[:, :-1]

    scaler = StandardScaler()
    returns = scaler.fit_transform(returns)

    if np.isnan(returns).any() or np.isinf(returns).any():
        print("Warning: NaN or Inf found in returns")
        returns = np.nan_to_num(returns)

    if returns.shape[0] < num_states:
        print("Warning: not enough data to train the model")
        return currentPos

    try:
        model.fit(returns.T)
    except ValueError as e:
        print("Error in fitting the model:", str(e))
        return currentPos

    if not np.allclose(model.transmat_.sum(axis=1), 1):
        print("Warning: transmat_ rows do not sum to 1")
        return currentPos

    hidden_states = model.predict(returns.T)

    # 调整交易位置
    # 我们将市场状态分为4种：强势上升、弱势上升、弱势下降、强势下降
    # 我们根据预测的市场状态进行相应的买入和卖出
    if hidden_states[-1] in [0, 1]:  # 预测为上升趋势
        currentPos = np.ones(nins) * 2000  # 全部买入
    elif hidden_states[-1] == 2:  # 预测为弱势下降
        currentPos = np.ones(nins) * -1000  # 部分卖出
    elif hidden_states[-1] == 3:  # 预测为强势下降
        currentPos = np.ones(nins) * -2000  # 全部卖出
    currentPos = f.clip_position(currentPos, prcSoFar)
    return currentPos


def loadPrices(fn):
    global nt, nInst
    # df=pd.read_csv(fn, sep='\s+', names=cols, header=None, index_col=0)
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return (df.values).T


if __name__ == "__main__":
    prcHist = loadPrices("prices.txt")
    pos = getMyPosition(prcHist)
    print(pos)
