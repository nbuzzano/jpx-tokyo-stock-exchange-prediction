
from models.utils import Feature
from models.raw_data import train_prices

feature_securities_code = Feature("securities_code", train_prices["SecuritiesCode"], 1)
feature_open = Feature("open", train_prices["Open"], 1)
feature_high = Feature("high", train_prices["High"], 1)
feature_low = Feature("low", train_prices["Low"], 1)
feature_close = Feature("close", train_prices["Close"], 1)
feature_volume = Feature("volume", train_prices["Volume"], 1)
