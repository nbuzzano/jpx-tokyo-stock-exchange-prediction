# export PYTHONPATH="${PYTHONPATH}:/home/nbuzzano/repositories/jpx-tokyo-stock-exchange-prediction"
# https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c

from math import sqrt

from models.raw_data import train_prices
from models.utils import save_experiment, timer, build_dataframe
from features.sector_code_17 import feature_17_sector_code
from features.sector_code_33 import feature_33_sector_code
from features.date import feature_date
# from features.volatility import (
# 	feature_realized_volatility,
# 	feature_parkinson_volatility,
# 	feature_garman_klass_volatility,
# 	feature_roger_satchell_volatility,
# 	feature_yang_zhang_volatility,
# 	feature_garkla_yangzh_volatility,
# )
from features.raw_features import (
	feature_securities_code,
	feature_open,
	feature_high,
	feature_low,
	feature_close,
	feature_volume,
)

from lightgbm import Dataset, train
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error #, accuracy_score, mean_squared_error, roc_auc_score,
from sklearn.pipeline import Pipeline # !!!

# https://github.com/nbuzzano/7506-zonaprop/blob/master/machine-learning/modelos/lightGBM.ipynb
# env = jpx_tokyo_market_prediction.make_env()
# iter_test = env.iter_test()


def prepare_lgbm_datasets(x_train, x_test, y_train, y_test):
    lgb_train = Dataset(x_train,y_train)#,free_raw_data=False)
    lgb_eval = Dataset(x_test,y_test)#,free_raw_data=False)
    return lgb_train,lgb_eval


def train_model(lgb_train, lgb_eval, x_test):

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

	y_test_pred = booster.predict(x_test)

	return y_test_pred, booster

def start_experiment():
	features = [
		feature_securities_code,
		feature_open,
		feature_high,
		feature_low,
		feature_close,
		feature_volume,
		feature_date,
		feature_17_sector_code,
		feature_33_sector_code
	]

	y = train_prices["Target"].copy()
	x = build_dataframe(features)

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)

	lgb_train,lgb_eval = prepare_lgbm_datasets(X_train, X_test, y_train, y_test)
	y_test_predicted, trained_model = train_model(lgb_train, lgb_eval, X_test)
	y_test_target = y_test

	save_experiment(
		'light_gbm', trained_model, features,
		y_test_predicted, y_test_target,
		metrics = [
			(mean_absolute_error, "mean_absolute_error"), 
			(lambda x,y: sqrt(mean_absolute_error(x,y)), "RMSE"),
			# (accuracy_score, "accuracy_score"),
			# (mean_squared_error, "mean_squared_error"),
			# (roc_auc_score,"roc_auc_score")
		]
	)

start_experiment()

# submit()
# TODO: FALTA COMPARAR COMO TE PIDE LA COMPETENCIA (los 1eros 200 y las ultimas 200 stocks era?)
