from models.utils import Feature
from models.raw_data import stock_list, train_prices

import pandas as pd

train_prices = train_prices.copy()
df = pd.merge(train_prices, stock_list[["SecuritiesCode", "17SectorCode"]], on = "SecuritiesCode", how = "left")
df["17SectorCode"] = df["17SectorCode"].astype("int64")

feature_17_sector_code = Feature("17_sector_code", df["17SectorCode"], 1)
