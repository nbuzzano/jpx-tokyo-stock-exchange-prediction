# import numpy as np
import pandas as pd
import os

LOCAL_PATH = 'data/jpx-tokyo-stock-exchange-prediction'
KAGGLE_PATH = '../input/jpx-tokyo-stock-exchange-prediction'
# ROOT_PATH = KAGGLE_PATH
ROOT_PATH = LOCAL_PATH

# train_options = pd.read_csv(f"{ROOT_PATH}/train_files/options.csv")
# train_secondary = pd.read_csv(f"{ROOT_PATH}/train_files/secondary_stock_prices.csv")
# train_financials = pd.read_csv(f"{ROOT_PATH}/train_files/financials.csv")
supplemental_prices = pd.read_csv(f"{ROOT_PATH}/supplemental_files/stock_prices.csv")
# supplemental_options = pd.read_csv(f"{ROOT_PATH}/supplemental_files/options.csv")
# supplemental_financials = pd.read_csv(f"{ROOT_PATH}/supplemental_files/financials.csv")
stock_list = pd.read_csv(f"{ROOT_PATH}/stock_list.csv")

train_prices = pd.read_csv(f"{ROOT_PATH}/train_files/stock_prices.csv")
# train_prices = train_prices.sample(n=5, random_state=42)

supplemental_prices = None
if not supplemental_prices:
    supplemental_prices = pd.read_csv(f"{ROOT_PATH}/supplemental_files/stock_prices.csv")
