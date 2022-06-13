# https://www.kaggle.com/code/satoshidatamoto/jpx-volatility-features/notebook

import pandas as pd
import pandas as pd
from math import sqrt, log

from models.utils import Feature
from models.raw_data import (
    stock_list,
    train_prices,
    supplemental_prices
) 

# import matplotlib.pyplot as plt    
# plt.style.use('bmh')
# plt.rcParams['figure.figsize'] = [14, 8]  # width, height
# def plot_feature(feature, df, feat_name):
#     breakpoint()
#     try: plt.close()
#     except: pass   
#     df2 = df[-len(feat):].reset_index().set_index('date')
#     fig = plt.figure(figsize = (12, 6))
#     # fig, ax_left = plt.subplots(figsize = (12, 6))
#     ax_left = fig.add_subplot(111)
#     ax_left.set_facecolor('azure')    
#     ax_right = ax_left.twinx()
#     ax_left.plot(feature, color = 'crimson', label = feat_name)
#     ax_right.plot(df['Close'], color = 'darkgrey', label = "Price")
#     plt.legend()
#     plt.grid()
#     plt.xlabel('Time')
#     plt.title('3 month rolling %s vs close price' % (feat_name))
#     plt.show()

def load_training_data(asset_id = None):

    # WHICH YEARS TO INCLUDE? YES=1 NO=0
    INC2022 = 1
    INC2021 = 1
    INC2020 = 1
    INC2019 = 1
    INC2018 = 1
    INC2017 = 1
    INCSUPP = 1
    
    df_train = pd.concat([train_prices, supplemental_prices]) if INCSUPP else train_prices
    df_train = pd.merge(df_train, stock_list[['SecuritiesCode', 'Name']], left_on = 'SecuritiesCode', right_on = 'SecuritiesCode', how = 'left')
    df_train['date'] = pd.to_datetime(df_train['Date'])
    df_train['year'] = df_train['date'].dt.year
    if not INC2022: df_train = df_train.loc[df_train['year'] != 2022]
    if not INC2021: df_train = df_train.loc[df_train['year'] != 2021]
    if not INC2020: df_train = df_train.loc[df_train['year'] != 2020]
    if not INC2019: df_train = df_train.loc[df_train['year'] != 2019]
    if not INC2018: df_train = df_train.loc[df_train['year'] != 2018]
    if not INC2017: df_train = df_train.loc[df_train['year'] != 2017]
    # asset_id = 1301 # Remove before flight
    if asset_id is not None: df_train = df_train.loc[df_train['SecuritiesCode'] == asset_id]
    # df_train = df_train[:1000] # Remove before flight
    return df_train


train = load_training_data().sort_values('date') #.set_index("date")
print("All data loaded!")

train_data = train.copy()
train_data['date'] = pd.to_datetime(train_data['Date'])
df = train_data.loc[train_data['SecuritiesCode'] == 1301]
df = df[-500000:]

#----------------------------------------------------------------

def realized(close, N=240):
    rt = list(log(C_t / C_t_1) for C_t, C_t_1 in zip(close[1:], close[:-1]))
    rt_mean = sum(rt) / len(rt)
    return sqrt(sum((r_i - rt_mean) ** 2 for r_i in rt) * N / (len(rt) - 1))

feature = df['Close'].rolling(60).apply(realized).bfill()
feature_realized_volatility = Feature("realized_volatility", feature, 1)
# plot_feature(feature, df, 'realized volatility')

#----------------------------------------------------------------

def parkinson(high, low, N=240):
    sum_hl = sum(log(H_t / L_t) ** 2 for H_t, L_t in zip(high, low))
    return sqrt(sum_hl * N / (4 * len(high) *log(2)))

feature = df.rolling(60).apply(lambda x: parkinson(df.loc[x.index, 'High'], df.loc[x.index, 'Low'])).bfill()
feature_parkinson_volatility = Feature("parkinson_volatility", feature, 1)
# plot_feature(feature, df, 'parkinson volatility')

#----------------------------------------------------------------

def garman_klass(open, high, low, close, N=240):
    sum_hl = sum(log(H_t / L_t) ** 2 for H_t, L_t in zip(high, low)) / 2
    sum_co = sum(log(C_t / O_t) ** 2 for C_t, O_t in zip(close, open)) * (2 * log(2) - 1)
    return sqrt((sum_hl - sum_co) * N / len(close))

feature = df.rolling(60).apply(lambda x: garman_klass(df.loc[x.index, 'Open'], df.loc[x.index, 'High'], df.loc[x.index, 'Low'], df.loc[x.index, 'Close'])).bfill()
feature_garman_klass_volatility = Feature("garman_klass", feature, 1)
# plot_feature(feature, df, 'garman klass')

#----------------------------------------------------------------

def roger_satchell(open, high, low, close, N=240):
    sum_ohlc = sum(log(H_t / C_t) * log(H_t / O_t) + log(L_t / C_t) * log(L_t / O_t) for O_t, H_t, L_t, C_t in zip(open, high, low, close))
    return sqrt(sum_ohlc * N / len(close))

feature = df.rolling(60).apply(lambda x: roger_satchell(df.loc[x.index, 'Open'], df.loc[x.index, 'High'], df.loc[x.index, 'Low'], df.loc[x.index, 'Close'])).bfill()
feature_roger_satchell_volatility = Feature("roger_satchell", feature, 1)
# plot_feature(feature, df, 'roger satchell')

#----------------------------------------------------------------

def yang_zhang(open, high, low, close, N=240):
    oc = list(log(O_t / C_t_1) for O_t, C_t_1 in zip(open[1:], close[:-1]))
    n = len(oc)
    oc_mean = sum(oc) / n
    oc_var = sum((oc_i - oc_mean) ** 2 for oc_i in oc) * N / (n - 1)   
    co = list(log(C_t / O_t) for O_t, C_t in zip(open[1:], close[1:]))
    co_mean = sum(co) / n
    co_var = sum((co_i - co_mean) ** 2 for co_i in co) * N / (n - 1)    
    rs_var = (roger_satchell(open[1:], high[1:], low[1:], close[1:])) ** 2    
    k = 0.34 / (1.34 + (n +1) / (n - 1))    
    return sqrt(oc_var + k * co_var + (1-k) * rs_var)

feature = df.rolling(60).apply(lambda x: yang_zhang(df.loc[x.index, 'Open'], df.loc[x.index, 'High'], df.loc[x.index, 'Low'], df.loc[x.index, 'Close'])).bfill()
feature_yang_zhang_volatility = Feature("yang_zhang", feature, 1)
# plot_feature(feature, df, 'yang zhang')

#----------------------------------------------------------------

def garkla_yangzh(open, high, low, close, N=240):
    sum_oc_1 = sum(log(O_t / C_t_1) ** 2 for O_t, C_t_1 in zip(open[1:], close[:-1]))
    sum_hl = sum(log(H_t / L_t) ** 2 for H_t, L_t in zip(high[1:], low[1:])) / 2
    sum_co = sum(log(C_t / O_t) ** 2 for C_t, O_t in zip(close[1:], open[1:])) * (2 * log(2) - 1)
    return sqrt((sum_oc_1 + sum_hl - sum_co) * N / (len(close) - 1))

feature = df.rolling(60).apply(lambda x: garkla_yangzh(df.loc[x.index, 'Open'], df.loc[x.index, 'High'], df.loc[x.index, 'Low'], df.loc[x.index, 'Close'])).bfill()
feature_garkla_yangzh_volatility = Feature("garkla_yangzh", feature, 1)
# plot_feature(feature, df, 'garkla yangzh')
