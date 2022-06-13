from models.utils import Feature
from models.raw_data import train_prices

import pandas as pd

train_prices = train_prices.copy()
train_prices['Date'] = pd.to_datetime(train_prices['Date'])
train_prices['Date'] = train_prices['Date'].dt.strftime("%Y%m%d").astype(int)
	
feature_date = Feature("date", train_prices["Date"], 1)
		