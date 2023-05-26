import numpy as np
import pandas as pd
import argparse
import os.path

from split.dataset_splitter import create_l1o_split, create_adjusted_time_based_split
from split.rules import create_sequential_rule_pairs, calculate_rule_overlap

def rule_overlap(full_path:str, train_path:str, test_path:str, session_key:str='SessionId', item_key:str='ItemId', time_key:str='Time') -> pd.DataFrame:
    """Calculates the A->B rule overlaps for different techniques on the dataset.

    Args:
        full_path (str): Path to the full dataset
        train_path (str): Path to the train dataset
        test_path (str): Path to the test dataset
        session_key (str, optional): Column name of the sessin ID. Defaults to 'SessionId'.
        item_key (str, optional): Column name of the item ID. Defaults to 'ItemId'.
        time_key (int, optional): Column name of the timestamp. Defaults to 'Time'.

    Returns:
        pd.DataFrame: DataFrame containing the results: dataset_name, method, overlap
    """
    #NOTE: It is assumed, that all 3 dataset contain sessions with at least 2 events 
    result = pd.DataFrame({"dataset_name":[], "method":[], "overlap":[]})
    dataset_name = os.path.split(full_path)[-1].split('_')[0]
    data_full = pd.read_csv(full_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    
    print("Calculating overlap for 'leave-one-out' technique")
    train_l1o, test_l1o = create_l1o_split(data_full, duplicate_penultimate_event=True)
    train_rules_l1o = create_sequential_rule_pairs(train_l1o)
    test_rules_l1o = create_sequential_rule_pairs(test_l1o)
    overlap_l1o = calculate_rule_overlap(test_rules_l1o, train_rules_l1o)
    train_events_l1o = len(train_l1o)
    test_events_l1o = len(test_l1o)-test_l1o.loc[:,session_key].nunique()
    print(f"\ttrain shape: {train_l1o.shape}", f"test shape: {test_l1o.shape}", f"test events: {test_events_l1o}")
    print(f"\ttrain_ratio: {train_events_l1o/len(data_full):.4f}", f"test_ratio: {test_events_l1o/len(data_full):.4f}")
    print(f"\toverlap: {overlap_l1o:.8f}")
    result.loc[0] = [dataset_name, "Leave-one-out", overlap_l1o]

    print(f"Calculating overlap for 'timebased split' technique, with test size adjusted to l1o test size {test_events_l1o/len(data_full):.4f}")
    train_tb, test_tb = create_adjusted_time_based_split(data_full, test_events=test_events_l1o)
    train_rules_tb = create_sequential_rule_pairs(train_tb)
    test_rules_tb = create_sequential_rule_pairs(test_tb)
    overlap_tb = calculate_rule_overlap(test_rules_tb, train_rules_tb)
    train_events_tb = len(train_tb)
    test_events_tb = len(test_tb)-test_tb.loc[:,session_key].nunique()
    print(f"\ttrain shape: {train_tb.shape}", f"test shape: {test_tb.shape}", f"test events: {test_events_tb}")
    print(f"\ttrain_ratio: {train_events_tb/len(data_full):.4f}", f"test_ratio: {test_events_tb/len(data_full):.4f}")
    print(f"\toverlap: {overlap_tb:.8f}")
    result.loc[1] = [dataset_name, "Time based", overlap_tb]

    print(f"Calculating overlap for 'leave-one-out with timebased split' technique")
    train = pd.read_csv(train_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    test = pd.read_csv(test_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    _, test_l1o_tb = create_l1o_split(test, force_test=True, duplicate_penultimate_event=True)
    train_rules = create_sequential_rule_pairs(train)
    test_rules_l1o_tb = create_sequential_rule_pairs(test_l1o_tb)
    overlap_l1o_tb = calculate_rule_overlap(test_rules_l1o_tb, train_rules)
    test_events_l1o_tb = len(test_l1o_tb)-test_l1o_tb.loc[:,session_key].nunique()
    print(f"\ttrain shape: {train.shape}", f"test shape: {test_l1o_tb.shape}", f"test events: {test_events_l1o_tb}")
    print(f"\toverlap: {overlap_l1o_tb:.8f}")
    result.loc[2] = [dataset_name, "Leave-one-out: time split", overlap_l1o_tb]
    
    print(f"Calculating overlap for 'leave-one-out with random split' technique")
    n_session_sample = test_l1o_tb.loc[:, session_key].nunique()
    train_l1o_random, test_l1o_random = create_l1o_split(data_full, force_test=True, duplicate_penultimate_event=True, n_session_sample=n_session_sample)
    train_rules_l1o_random = create_sequential_rule_pairs(train_l1o_random)
    test_rules_l1o_random = create_sequential_rule_pairs(test_l1o_random)
    overlap_l1o_random = calculate_rule_overlap(test_rules_l1o_random, train_rules_l1o_random)
    test_events_l1o_random = len(test_l1o_random) - test_l1o_random.loc[:, session_key].nunique()
    print(f"\ttrain shape: {train_l1o_random.shape}", f"test shape: {test_l1o_random.shape}", f"test events: {test_events_l1o_random}")
    print(f"\toverlap: {overlap_l1o_random:.8f}")
    result.loc[3] = [dataset_name, "Leave-one-out: random", overlap_l1o_random]

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_path', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    args = parser.parse_args()
    result = rule_overlap(args.full_path, args.train_path, args.test_path)
    print(result)