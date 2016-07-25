import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import functools as ft
from collections import Callable


def cum_ret(x, y):
    return x * (1+y)


def reduce_cum_ret(y):
    return ft.reduce(cum_ret, y, 10000.0)


class SmAvg(Callable):
    def __init__(self, c):
        self.c_ = c

    def __call__(self, x, y):
        return self.c_*y+(1-self.c_)*x


class ReduceSmAvg(Callable):
    def __init__(self, xx, f):
        self.xx_ = xx
        self.f_ = f

    def __call__(self, y):
        return ft.reduce(self.f_, y, self.xx_)


def create_features(fn):
    data = pd.read_csv(fn, dtype={'Date': datetime.date, 'SPX': np.float64},
                       index_col='Date',
                       infer_datetime_format=True)  # Date, SPX

    print('after CSV loading, now={0}'.format(datetime.datetime.now()))
    data['Performance'] = data['SPX'] / data['SPX'].shift(1) - 1.0  # how to specify type?
    data.set_value('01/02/2008', 'Performance', 0.0)
    # print(data.head())
    # print(data.dtypes)

    data['N10000'] = data['Performance'].expanding().apply(reduce_cum_ret)
    data['CumReturns'] = data['N10000'] / 10000.0 - 1.0

    data['SMA_5'] = data['SPX'].rolling(window=5).mean()  # pd.rolling_mean(data['SPX'], 5)[4:]

    data['SMA_25'] = data['SPX'].rolling(window=25).mean()
    data['Change'] = data['SPX'] - data['SPX'].shift(1)
    data['Change'].fillna(0)
    data.set_value('01/02/2008', 'Change', 0.0)

    data['Move_Up'] = data['Change'].copy()
    data.loc[data.Change < 0.0, 'Move_Up'] = 0.0  # todo: can we combine deep-copy-n-modify into one statement

    data['Move_Down'] = -data['Change']
    data.loc[data.Move_Down < 0.0, 'Move_Down'] = 0.0

    data['Avg_Up'] = data['Move_Up'].rolling(14).mean().fillna(0.0)
    data['Avg_Down'] = data['Move_Down'].rolling(14).mean().fillna(0.0)

    const_p1 = 1.0 / 14.0
    sm_offset = 14
    data['Smoothed_Avg_Up'] = np.zeros(data.shape[0])
    data_move_up = data['Move_Up']
    data['Smoothed_Avg_Up'][sm_offset:] = data_move_up[sm_offset:].expanding().apply(
        ReduceSmAvg(data.at['01/22/2008', 'Avg_Up'], SmAvg(const_p1)))

    data['Smoothed_Avg_Down'] = np.zeros(data.shape[0])
    data_move_dn = data['Move_Down']
    data['Smoothed_Avg_Down'][sm_offset:] = data_move_dn[sm_offset:].expanding().apply(
        ReduceSmAvg(data.at['01/22/2008', 'Avg_Down'], SmAvg(const_p1)))
    data['RS'] = data['Smoothed_Avg_Up'] / data['Smoothed_Avg_Down']
    data['RS'].fillna(0.0)
    data['RSI_14'] = 100.0 - 100.0 / (1.0 + data['RS'])

    # implement shift-1-day logic into feature
    # featureData = pd.concat([data['SMA_5'], data['SMA_25'], data['RSI_14']], axis= 1)

    out = data[['SMA_5', 'SMA_25', 'RSI_14']]

    out.shift(1)
    return out, data['Performance']


if __name__ == '__main__':
    fn = 'd:/dev/tinyp/PositionSizing/Excel/spx.csv'
    df, perf_series = create_features(fn)

    df['Signal'] = np.select([df['SMA_5'] > df['SMA_25']], [1], default=-1)

    df['Signal'][:25] = 0
    df['01/02/2008':'01/22/2008']['RSI_14'] = 0
    rsi = df['RSI_14']
    df['S_0_30'] = np.where(df['RSI_14'] < 30, 0.3, 0)
    df['S_30_50'] = np.where( (rsi < 50) & (rsi > 30), 1, 0)  # somehow 30 < rsi < 50 condition will throw
    df['S_50_70'] = np.where( (rsi < 70) & (rsi > 50), 1, 0)
    df['S_70_100'] = np.where(70 < rsi, 0.3, 0)

    df['S_sum'] = df['S_0_30'] + df['S_30_50'] + df['S_50_70'] + df['S_70_100']

    df['Performance'] = perf_series
    df['Strat_Daily_Return'] = df['Performance'] * df['Signal'] * df['S_sum']
    df['Strat_N10000'] = df['Strat_Daily_Return'].expanding().apply(reduce_cum_ret)
    df['Strat_CumReturns'] = df['Strat_N10000'] / 10000.0 - 1.0
    print('after running strat, now={0}'.format(datetime.datetime.now()))
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.expand_frame_repr', False)

    print('Head')
    print(df.head(40))
    print('Tail')
    print(df.tail(20))

    input('Press any key to see graph')

    plt.figure()
    df[['Performance', 'Strat_CumReturns']].plot()
    plt.interactive(False)
    plt.show()
