import pickle
import os
import pandas as pd
from datetime import datetime


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
    print('Saving model into a pickle')	
    with open(f'{target_dir}/{model_name}_pickle.pkl','wb') as f:
	    pickle.dump(model, f)


def save_metrics(y_test, y_test_pred, metric_list, target_dir):
    print('Saving metrics')	
    with open(f'{target_dir}/metrics.txt', 'a') as out:
        for metric_function, metric_name in metric_list:
            metric_value = metric_function(y_test, y_test_pred)
            out.write(f"{metric_name}: {metric_value}" + '\n')


def save_experiment(model_name, model, y_test_predicted, y_test_target, metrics):

    timestamp = datetime.now()
    target_dir = f'predictions/experiments/{model_name}/{timestamp}/'
    
    print('\nSaving experiment')	
    create_target_dir(target_dir)
    save_prediction(y_test_predicted, y_test_target, model_name, target_dir)
    save_pickle(model, model_name, target_dir)
    save_metrics(y_test_target, y_test_predicted, metrics, target_dir)
    print('Done.')	
    