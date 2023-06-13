import os
import pandas as pd
from session_rec.algorithms.knn import vsknn
from session_rec.evaluation import evaluation_last, evaluation
from session_rec.evaluation.metrics.accuracy import HitRate

def train_test_vsknn_models(experiments, setups, m=None):
    for dataset_key, dataset_paths in experiments.items():
        for model_key, params in setups[dataset_key].items():    
            train_path = dataset_paths["train_path"]
            test_path = dataset_paths.get("test_path")
            train_data = pd.read_csv(train_path, sep='\t')
            train_data = train_data.sort_values(by=["SessionId", "Time", "ItemId"], ascending=True)
            test_data = pd.read_csv(test_path, sep='\t')
            test_data = test_data.sort_values(by=["SessionId", "Time", "ItemId"], ascending=True)

            train_unique_items = train_data["ItemId"].unique()
            test_data = test_data[test_data["ItemId"].isin(train_unique_items)]
            session_length_test = test_data.groupby("SessionId").size()
            test_data = test_data[test_data["SessionId"].isin(session_length_test[session_length_test > 1].index)]
            print(f"{dataset_key} {model_key}")
            print("\t", "TRAIN, scoreing ALL")
            model = vsknn.VMContextKNN(**params)
            model.fit(train_data)
            print("\t", "EVAL, scoreing ALL")
            metrics = [HitRate(cutoff) for cutoff in m]
            res = evaluation.evaluate_sessions(pr=model, metrics=metrics, test_data=test_data, train_data=train_data)
            for r in res:
                print("\t", r[0], f"{r[1]:.6f}")