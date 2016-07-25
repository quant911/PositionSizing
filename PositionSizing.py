import pandas as pd
import numpy as np
import datetime
import functools as ft

def cumRet(x, y):
    return x * (1+y)

def foldCumRet(x):
    return ft.reduce(cumRet, x, 10000.0)

if __name__ == '__main__':
    data = pd.read_csv('d:/dev/tinyp/PositionSizing/Excel/spx.csv', dtype={'Date': datetime.date, 'SPX': np.float64},
                       index_col='Date',
                       infer_datetime_format=True)  # Date, SPX
    data['Performance'] = data['SPX']/ data['SPX'].shift(1)  - 1.0  # how to specify type?
    data.set_value('01/02/2008', 'Performance', 0.0)
    print(data.head())
    print(data.dtypes)

    data['N10000'] = data['Performance'].expanding().apply(foldCumRet)
    data['CumReturns'] = data['N10000'] / 10000.0 - 1.0

    data['SMA_5'] = data['SPX'].rolling(window=5).mean() # pd.rolling_mean(data['SPX'], 5)[4:]
    print(data.head(10))

    data['SMA_25'] = data['SPX'].rolling(window=25).mean()
    data['Change'] = data['SPX'] - data['SPX'].shift(1)
    data['Change'].fillna(0)
    data.set_value('01/02/2008', 'Change', 0.0)
    # try data.ix(data.Date='01/02/2008', 'Change') = 0.0
    print(data.head(40))
    idx_up = data['Change'] >0
    # data['Move_Up'] = data['Change'].map(lambda c: np.floor(c, 0.0)) # np.floor(data['Change'], 0)
    # !!!!
    data['Move_Up'] = data['Change'].copy()
    data.loc[data.Change<0.0, 'Move_Up'] = 0.0 # todo: can we combine deep-copy-n-modify into one statement

    data['Move_Down'] = -data['Change']
    data.loc[data.Move_Down <0.0, 'Move_Down'] = 0.0

    data['Avg_Up'] = data['Move_Up'].rolling(14).mean().fillna(0.0)
    data['Avg_Down'] = data['Move_Down'].rolling(14).mean().fillna(0.0)

    const_p1 = 1.0/14.0
    data['Smoothed_Avg_Up'] = data['Move_Up'] * const_p1 + data['Avg_Up'].shift(1) * (1.0 - const_p1) # we need fold and access two cols!
    v01 = data.at['01/23/2008', 'Move_Up']
    v02 = data.at['01/22/2008', 'Avg_Up']
    v0 =  v01 * const_p1 +  v02 * (1.0 - const_p1)
    print('MoveUp={0}, AvgUp={1}, Sm={2}'.format(v01, v02, v0))
    # v0 = data.ix('01/22/2008', 'Move_Up') * const_p1 + data.get_value('01/18/2008', 'Avg_Up') * (1.0 - const_p1)
    data['Smoothed_Avg_Down'] = data['Move_Down'] * const_p1 + data['Avg_Down'].shift(1) * (1.0 - const_p1)
    data['RS'] = data['Smoothed_Avg_Up'] / data['Smoothed_Avg_Down']
    data['RSI_14'] = 100.0 - 100.0 / (1.0 + data['RS'])
    data['Signal'] = np.select([ data['SMA_5']>data['SMA_25']], [1], default=-1)

    data['Position_Sizing_0_30'] = np.where(data['RSI_14'].shift(1) < 30, 0.3, 0)
#    data['Position_Sizing_30_50'] = np.where(data['RSI_14'].shift(1) > 30 & data['RSI_14'].shift(1) < 50, 1, 0)
    data['Position_Sizing_30_50'] = data['Position_Sizing_0_30'].copy() # fake!!!
    data['Position_Sizing_50_70'] = data['Position_Sizing_0_30'].copy()
    data['Position_Sizing_70_100'] = data['Position_Sizing_0_30'].copy()

    data['Position_Sizing_sum'] = data['Position_Sizing_0_30'] + data['Position_Sizing_30_50'] + data['Position_Sizing_50_70'] + data['Position_Sizing_70_100']

    #data['Move_Up'] = [ np.floor(c, 0.0) for c in data['Change'] ]
    # data['Move_Down'] = data['Change'].apply(lambda c: -np.ceil(c, 0.0) ) # np.fabs(np.ceil(data['Change'], 0))
    # data['Move_Up'] = data[data['Change'] > 0].fillna(0)

    print(data.head(40))