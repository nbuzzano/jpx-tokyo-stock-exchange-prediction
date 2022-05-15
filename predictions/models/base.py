# import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_options = pd.read_csv("data/jpx-tokyo-stock-exchange-prediction/train_files/options.csv")
train_secondary = pd.read_csv("data/jpx-tokyo-stock-exchange-prediction/train_files/secondary_stock_prices.csv")
train_financials = pd.read_csv("data/jpx-tokyo-stock-exchange-prediction/train_files/financials.csv")
supplemental_prices = pd.read_csv("data/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv")
supplemental_options = pd.read_csv("data/jpx-tokyo-stock-exchange-prediction/supplemental_files/options.csv")
supplemental_financials = pd.read_csv("data/jpx-tokyo-stock-exchange-prediction/supplemental_files/financials.csv")
stock_list = pd.read_csv("data/jpx-tokyo-stock-exchange-prediction/stock_list.csv")

train_prices = pd.read_csv("data/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv")
train_prices = train_prices.sample(n=5, random_state=42)
