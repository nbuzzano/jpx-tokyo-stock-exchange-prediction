import pickle
import pandas as pd
from datetime import datetime
import string


class Feature:
	version = string
	feature = pd.DataFrame
	name = string

	def __init__(self, name, feature, version):
		self.name = name
		self.version = version
		self.feature = feature


def build_dataframe(features):
	df = pd.DataFrame()
	for f in features:
		df[f.name] = f.feature
	return df


def timer(start_time=None):
	if not start_time:
		start_time = datetime.now()
		return start_time
	elif start_time:
		thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
		tmin, tsec = divmod(temp_sec, 60)
		print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def create_target_dir(target_dir):
    from pathlib import Path 
    Path(target_dir).mkdir(parents=True, exist_ok=True)


def save_prediction(y_predicted, y_target, model_name, target_dir):
    print('Saving predictions')	
    submit = pd.DataFrame({'prediction': y_predicted, 'target': y_target})
    submit.to_csv(f'{target_dir}/submit-'+model_name+'.csv', index=False)


def save_pickle(model, model_name, target_dir):
    print('Saving model as pickle')	
    with open(f'{target_dir}/{model_name}_pickle.pkl','wb') as f:
	    pickle.dump(model, f)


def save_metrics(y_test, y_test_pred, metric_list, target_dir):
    print('Saving metrics')	
    with open(f'{target_dir}/metrics.txt', 'a') as out:
        for metric_function, metric_name in metric_list:
            metric_value = metric_function(y_test, y_test_pred)
            metric_log = f"{metric_name}: {metric_value}"
            out.write(f"{metric_log}\n")
            print(f"    {metric_log}")


def save_features(features, target_dir):
    print('Saving features')	
    with open(f'{target_dir}/features.txt', 'a') as out:
        for f in features:
            out.write(f"feature name: {f.name}, version: {f.version}" + '\n')
    pass


def save_experiment(model_name, model, features, y_test_predicted, y_test_target, metrics):

    timestamp = datetime.now()
    target_dir = f'predictions/experiments/{model_name}/{timestamp}/'
    
    print(f'\nSaving experiment into {target_dir}')	
    create_target_dir(target_dir)
    save_prediction(y_test_predicted, y_test_target, model_name, target_dir)
    save_features(features, target_dir)
    save_pickle(model, model_name, target_dir)
    save_metrics(y_test_target, y_test_predicted, metrics, target_dir)
    print('Done.')


import data."jpx-tokyo-stock-exchange-prediction".jpx_tokyo_market_prediction

def submit(df):
    _SUBMIT_ENABLED = False
    if not _SUBMIT_ENABLED:
        print("Submit not enabled")
        pass
    
    env = jpx_tokyo_market_prediction.make_env()
    iter_test = env.iter_test()
    for (prices, _, _, _, _, submission) in iter_test:
        # prices["Spread"] = prices.High - prices.Low
        # prices["Rank"] = prices.groupby("Date")["Spread"].rank(method="first", ascending=False, na_option="bottom").astype(int) - 1
        submission = submission.drop(columns=["Rank"])
        submission = submission.merge(prices[["SecuritiesCode", "Rank"]], on="SecuritiesCode", how="inner")
        env.predict(submission)