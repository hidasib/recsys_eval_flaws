import numpy as np
import pandas as pd
import os

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