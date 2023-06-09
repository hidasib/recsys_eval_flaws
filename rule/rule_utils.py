import numpy as np
import pandas as pd
import os
import sys
sys.path.append("../split")
from split.dataset_splitter import create_l1o_split, create_adjusted_time_based_split

def create_sequential_rule_pairs(df, session_key='SessionId', item_key='ItemId', time_key='Time'):
    df_temp = df.copy()
    shifted_itemids = df_temp.loc[:, item_key].iloc[1:].values
    df_temp = df_temp.iloc[:-1]
    df_temp["shifted_item_id"] = shifted_itemids
    session_end_mask = np.concatenate((df_temp.loc[:, session_key].values[:-1] != df_temp.loc[:, session_key].values[1:], [True]))
    df_temp = df_temp[~session_end_mask]
    rules = df_temp.loc[:, item_key] + '_' + df_temp.shifted_item_id
    return rules

def calculate_rule_overlap(rules_test, rules_train):
    rules_test_unique = set(rules_test)
    rules_train_unique = set(rules_train)
    return len(rules_test_unique.intersection(rules_train_unique)) / len(rules_test_unique)

def calculate_new_rules(full_path:str, time_units:str='ms', session_key:str='SessionId', item_key:str='ItemId', time_key:str='Time') -> pd.DataFrame:
    """Calculates the ratio of new A->B sequential rules compared to the amount of unique rules on the same day.  

    Args:
        full_path (str): Path to the full dataset 
        time_units (str, optional): The units of the unix timestamp seconds ('s') or miliseconds ('ms'). Defaults to 'ms'.
        session_key (str, optional): Column name of the sessin ID. Defaults to 'SessionId'.
        item_key (str, optional): Column name of the item ID. Defaults to 'ItemId'.
        time_key (int, optional): Column name of the timestamp. Defaults to 'Time'.

    Returns:
        pd.DataFrame: ratio of new unique rules compared to the unqiue rules on a given day  
    """
    print(f"Calculating daily new rules for: {full_path}")
    dataset_name = os.path.split(full_path)[-1].split('_')[0]
    result_file_name = os.path.join("data", "results", "rule_new", f"{dataset_name}_new_rules.tsv")
    if os.path.isfile(result_file_name):
        print(f"\tSKIPPING, file exists: {result_file_name}")
        return
    data_full = pd.read_csv(full_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    data_full.Time = pd.to_datetime(data_full.Time, unit=time_units)
    timeframed = data_full.groupby([pd.Grouper(key=time_key, freq='D')], as_index=False)
    new_rules_ratios = []
    all_rules = set()
    for start_time, data_chunk in timeframed:
        rules = create_sequential_rule_pairs(data_chunk, session_key, item_key, time_key)
        overlap_ratio = calculate_rule_overlap(rules, all_rules)
        new_rules_ratios.append(1 - overlap_ratio)
        all_rules.update(rules)
    new_rules_ratio = pd.DataFrame(new_rules_ratios)
    new_rules_ratio.to_csv(result_file_name, sep='\t', index=False, header=False)

def rule_overlap(full_path:str, train_path:str, test_path:str, session_key:str='SessionId', item_key:str='ItemId', time_key:str='Time') -> pd.DataFrame:
    """Calculates the A->B rule overlaps between test and train set for different splitting techniques.

    Args:
        full_path (str): Path to the full dataset
        train_path (str): Path to the train dataset (original time based split)
        test_path (str): Path to the test dataset (original time based split)
        session_key (str, optional): Column name of the sessin ID. Defaults to 'SessionId'.
        item_key (str, optional): Column name of the item ID. Defaults to 'ItemId'.
        time_key (int, optional): Column name of the timestamp. Defaults to 'Time'.

    Returns:
        pd.DataFrame: DataFrame containing the results: dataset_name, method, overlap
    """
    dataset_name = os.path.split(full_path)[-1].split('_')[0]
    print(f"Calculating rule overlap for: {dataset_name}")
    result_file_name = os.path.join("data", "results", "rule_overlap", f"{dataset_name}_rule_overlap.tsv")
    if os.path.isfile(result_file_name):
        print(f"\tSKIPPING, file exists: {result_file_name}")
        return
    #NOTE: It is assumed, that all 3 datasets contain sessions with at least 2 events 
    result = pd.DataFrame({"dataset_name":[], "method":[], "overlap":[]})
    data_full = pd.read_csv(full_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    
    print("\tCalculating overlap for 'leave-one-out' technique")
    train_l1o, test_l1o = create_l1o_split(data_full, duplicate_penultimate_event=True)
    train_rules_l1o = create_sequential_rule_pairs(train_l1o)
    test_rules_l1o = create_sequential_rule_pairs(test_l1o)
    overlap_l1o = calculate_rule_overlap(test_rules_l1o, train_rules_l1o)
    train_events_l1o = len(train_l1o)
    test_events_l1o = len(test_l1o)-test_l1o.loc[:,session_key].nunique()
    print(f"\t\ttrain shape: {train_l1o.shape}", f"test shape: {test_l1o.shape}", f"test events: {test_events_l1o}")
    print(f"\t\ttrain_ratio: {train_events_l1o/len(data_full):.4f}", f"test_ratio: {test_events_l1o/len(data_full):.4f}")
    print(f"\t\toverlap: {overlap_l1o:.8f}")
    result.loc[0] = [dataset_name, "Leave-one-out", overlap_l1o]

    print(f"\tCalculating overlap for 'timebased split' technique, with test size adjusted to l1o test size {test_events_l1o/len(data_full):.4f}")
    train_tb, test_tb = create_adjusted_time_based_split(data_full, test_events=test_events_l1o)
    train_rules_tb = create_sequential_rule_pairs(train_tb)
    test_rules_tb = create_sequential_rule_pairs(test_tb)
    overlap_tb = calculate_rule_overlap(test_rules_tb, train_rules_tb)
    train_events_tb = len(train_tb)
    test_events_tb = len(test_tb)-test_tb.loc[:,session_key].nunique()
    print(f"\t\ttrain shape: {train_tb.shape}", f"test shape: {test_tb.shape}", f"test events: {test_events_tb}")
    print(f"\t\ttrain_ratio: {train_events_tb/len(data_full):.4f}", f"test_ratio: {test_events_tb/len(data_full):.4f}")
    print(f"\t\toverlap: {overlap_tb:.8f}")
    result.loc[1] = [dataset_name, "Time based", overlap_tb]

    print(f"\tCalculating overlap for 'leave-one-out with timebased split' technique")
    train = pd.read_csv(train_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    test = pd.read_csv(test_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    _, test_l1o_tb = create_l1o_split(test, force_test=True, duplicate_penultimate_event=True)
    train_rules = create_sequential_rule_pairs(train)
    test_rules_l1o_tb = create_sequential_rule_pairs(test_l1o_tb)
    overlap_l1o_tb = calculate_rule_overlap(test_rules_l1o_tb, train_rules)
    test_events_l1o_tb = len(test_l1o_tb)-test_l1o_tb.loc[:,session_key].nunique()
    print(f"\t\ttrain shape: {train.shape}", f"test shape: {test_l1o_tb.shape}", f"test events: {test_events_l1o_tb}")
    print(f"\t\toverlap: {overlap_l1o_tb:.8f}")
    result.loc[2] = [dataset_name, "Leave-one-out: time split", overlap_l1o_tb]
    
    print(f"\tCalculating overlap for 'leave-one-out with random split' technique")
    n_session_sample = test_l1o_tb.loc[:, session_key].nunique()
    train_l1o_random, test_l1o_random = create_l1o_split(data_full, force_test=True, duplicate_penultimate_event=True, n_session_sample=n_session_sample)
    train_rules_l1o_random = create_sequential_rule_pairs(train_l1o_random)
    test_rules_l1o_random = create_sequential_rule_pairs(test_l1o_random)
    overlap_l1o_random = calculate_rule_overlap(test_rules_l1o_random, train_rules_l1o_random)
    test_events_l1o_random = len(test_l1o_random) - test_l1o_random.loc[:, session_key].nunique()
    print(f"\t\ttrain shape: {train_l1o_random.shape}", f"test shape: {test_l1o_random.shape}", f"test events: {test_events_l1o_random}")
    print(f"\t\toverlap: {overlap_l1o_random:.8f}")
    result.loc[3] = [dataset_name, "Leave-one-out: random", overlap_l1o_random]

    result.to_csv(result_file_name, index=False, sep='\t')