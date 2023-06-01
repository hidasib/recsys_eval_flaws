import sys
sys.path.append("./session_rec/")
import pandas as pd
import numpy as np
from session_rec.algorithms.knn import vsknn
from session_rec.evaluation import evaluation, evaluation_last
from session_rec.evaluation.metrics.accuracy import HitRate
import argparse
import optuna
import json
import os

class Objective:
    def __init__(self, train_data, test_data, log_path, pr_class, pr_param_grid,  res_f_name, eval_method):
        self.pr_class = pr_class
        self.pr_param_grid = pr_param_grid
        self.train_data = train_data
        self.test_data = test_data
        self.res_f_name = res_f_name
        self.log_path = log_path
        self.best_best = {"trial": -1, "Recall@20": -1}
        self.eval_method = eval_method
        with open(self.log_path, 'w') as file:
            file.write('')
    
    def __call__(self, trial):
        for param_name, param_space in self.pr_param_grid.items():
            trial.suggest_categorical(param_name, param_space)
        local_config = trial.params
        model_local = self.pr_class(**local_config)
        model_local.fit(self.train_data)
        if self.eval_method == "all":
            res = evaluation.evaluate_sessions(pr=model_local, metrics=[HitRate(20)], test_data=test_data, train_data=train_data)
        elif self.eval_method == "last":
            res = evaluation_last.evaluate_sessions(pr=model_local, metrics=[HitRate(20)], test_data=test_data, train_data=train_data)
        metric = res[-1][1]

        if metric > self.best_best["Recall@20"]:
            self.best_best["trial"] = trial.number
            self.best_best["Recall@20"] = metric
        with open(self.log_path, 'a') as file:
            optim_log = {"trial": trial.number, "metric": metric, "params": local_config, "best": {"trial": self.best_best["trial"], "metric": metric}} 
            json.dump(optim_log, file)
        return metric

def get_model_class_param_grid(model_name:str):
    if model_name == "vsknn":
        model_param_grid = {
            "k": [50, 100, 500, 1000, 1500],
            "sample_size": [500, 1000, 2500, 5000, 10000],
            "weighting": ["same", "div", "linear", "quadratic", "log"],
            "weighting_score": ["same", "div", "linear", "quadratic", "log"], 
            "idf_weighting": [False, 1, 2, 5, 10],
            }
        model_class = vsknn.VMContextKNN
    else:
        raise ValueError(f"Invalid command line model arg: {model_name} model is not supported")
    return model_class, model_param_grid
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--eval_method', type=str, help="'all' or 'last', indicating wether to calculate metric for all test sessions, or only for the last event in the test sessions", default="all")
    parser.add_argument('--session_key', type=str, default="SessionId")
    parser.add_argument('--item_key', type=str, default="ItemId")
    parser.add_argument('--time_key', type=str, default="Time")
    parser.add_argument('--max_iters', type=int, default=100)
    args = parser.parse_args()

    model_class, model_param_grid = get_model_class_param_grid(args.model)
    
    train_data = pd.read_csv(args.train_path, sep='\t')
    train_data = train_data.sort_values(by=[args.session_key, args.time_key, args.item_key], ascending=True)
    test_data = pd.read_csv(args.test_path, sep='\t')
    test_data = test_data.sort_values(by=[args.session_key, args.time_key, args.item_key], ascending=True)
    train_unique_items = train_data[args.item_key].unique()
    test_data = test_data[test_data[args.item_key].isin(train_unique_items)]
    session_length_test = test_data.groupby(args.session_key).size()
    test_data = test_data[test_data[args.session_key].isin(session_length_test[session_length_test > 1].index)]
    
    _, fn = os.path.split(args.train_path)
    res_f_name = f"optuna_recall_{args.model}_maxiter{args.max_iters}_{fn[:-4]}"
    log_path = os.path.join("data", "results", "vskk_paropt", f"{res_f_name}.json")
    log_path_best = os.path.join("data", "results", "vskk_paropt", f"{res_f_name}_best.json")
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    objective = Objective(train_data, test_data, log_path=log_path, pr_class=model_class, pr_param_grid=model_param_grid, res_f_name=res_f_name, eval_method=args.eval_method)
    study.optimize(objective, n_trials=args.max_iters, n_jobs=1, show_progress_bar=True)
    
    result = {"model": args.model, "train_file":args.train_path, "test_file":args.test_path, "optim_best_recall": study.best_trial.value, "n_trials": args.max_iters, "best_params": study.best_trial.params}
    print(result)
    with open(log_path_best, 'w') as file:
        json.dump(result, file)