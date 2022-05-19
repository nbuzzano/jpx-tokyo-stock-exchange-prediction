# export PYTHONPATH="${PYTHONPATH}:/home/nbuzzano/repositories/jpx-tokyo-stock-exchange-prediction"
# https://towardsdatascience.com/how-to-fix-modulenotfounderror-and-importerror-248ce5b69b1c

from math import sqrt

from predictions.models.base import train_prices
from predictions.models.utils import save_experiment, timer, Feature, build_dataframe
from predictions.features.sector_code_17 import feature_17_sector_code
from predictions.features.sector_code_33 import feature_33_sector_code
from predictions.features.date import feature_date
from predictions.features.volatility import (
	feature_realized_volatility,
	feature_parkinson_volatility,
	feature_garman_klass_volatility,
	feature_roger_satchell_volatility,
	feature_yang_zhang_volatility,
	feature_garkla_yangzh_volatility,
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


features = [
	Feature("securities_code", train_prices["SecuritiesCode"], 1),
	Feature("open", train_prices["Open"], 1),
	Feature("high", train_prices["High"], 1),
	Feature("low", train_prices["Low"], 1),
	Feature("close", train_prices["Close"], 1),
	Feature("volume", train_prices["Volume"], 1),
	feature_date,
	feature_17_sector_code,
	feature_33_sector_code
]

y = train_prices["Target"].copy()
x = build_dataframe(features)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)

lgb_train,lgb_eval = prepare_lgbm_datasets(X_train, X_test, y_train, y_test)
y_test_predicted, trained_model = train_model(lgb_train,lgb_eval)
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

# submit()
# TODO: FALTA COMPARAR COMO TE PIDE LA COMPETENCIA (los 1eros 200 y las ultimas 200 stocks era?)
