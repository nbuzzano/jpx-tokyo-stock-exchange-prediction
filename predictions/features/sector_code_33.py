from predictions.models.utils import Feature
from predictions.models.base import stock_list, train_prices
import pandas as pd

train_prices = train_prices.copy()
df = pd.merge(train_prices, stock_list[["SecuritiesCode", "33SectorCode"]], on = "SecuritiesCode", how = "left")
df["33SectorCode"] = df["33SectorCode"].astype("int64")

feature_33_sector_code = Feature("33_sector_code", df["33SectorCode"], 1)
