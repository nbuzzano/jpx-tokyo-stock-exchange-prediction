# export PYTHONPATH="${PYTHONPATH}:/home/nbuzzano/repositories/jpx-tokyo-stock-exchange-prediction"
# https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c

from predictions.models.base import stock_list, train_prices
from predictions.models.utils import save_experiment, timer

import pandas as pd
from datetime import datetime
from math import sqrt

import lightgbm
from lightgbm import Dataset 
from lightgbm import train

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, mean_absolute_error
# https://github.com/nbuzzano/7506-zonaprop/blob/master/machine-learning/modelos/lightGBM.ipynb
from sklearn.pipeline import Pipeline # !!!

# env = jpx_tokyo_market_prediction.make_env()
# iter_test = env.iter_test()


def load_data(df):
	# TODO: refactorear esto, deberia crear las features en distintos archivos
	# para que no se arme bolsa de gatos e importar las features aca 
	# e ir apendeandolas, asi puedo mayor flexibilidad al armar set de features,
	# save_experiment tmb deberia guardar que set de features con su version corrio cada modelo.
	feature_cols = ["Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume", "17SectorCode"]
	
	_df = df[feature_cols].copy()
	
	_df = pd.merge(_df, stock_list[["SecuritiesCode", "17SectorCode"]], on = "SecuritiesCode", how = "left")
	_df["17SectorCode"] = _df["17SectorCode"].astype("int64")
	
	_df['Date'] = pd.to_datetime(_df['Date'])
	_df['Date'] = _df['Date'].dt.strftime("%Y%m%d").astype(int)
	
	target = df["Target"].copy()
	
	return _df, target


def prepare_data(x_train, x_test, y_train, y_test):
    lgb_train = Dataset(x_train,y_train)#,free_raw_data=False)
    lgb_eval = Dataset(x_test,y_test)#,free_raw_data=False)
    return lgb_train,lgb_eval


def train_model(lgb_train, lgb_eval):

	print('Start training...')
	start_time = timer()

	params = {
		'num_leaves':150,
		'max_depth':7,
		'learning_rate':.02,
		'max_bin':150,
		'num_iterations':100, #default = 100          
		'metric': ['auc', 'mae', 'rmse']
	}  

	booster = train(
		params=params,
		train_set=lgb_train,
		valid_sets=[lgb_eval],
		# verbose_eval=100,
	)

	timer(start_time)
	print('Training ended...')

	y_test_pred = booster.predict(X_test)

	return y_test_pred, booster


_train_prices = train_prices.sample(n=5, random_state=42)
x, y = load_data(_train_prices)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)
lgb_train,lgb_eval = prepare_data(X_train, X_test, y_train, y_test)

y_test_predicted, trained_model = train_model(lgb_train,lgb_eval)

save_experiment(
	'light_gbm', trained_model,
	y_test_predicted, y_test,
	metrics = [
        (mean_absolute_error, "mean_absolute_error"), 
        (lambda x,y: sqrt(mean_absolute_error(x,y)), "RMSE"),
        # (accuracy_score, "accuracy_score"),
        # (mean_squared_error, "mean_squared_error"),
        # (roc_auc_score,"roc_auc_score")
	]
)

# TODO: FALTA COMPARAR COMO TE PIDE LA COMPETENCIA (los 1eros 200 y las ultimas 200 stocks era?)