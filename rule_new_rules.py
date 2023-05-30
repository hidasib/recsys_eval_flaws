import numpy as np
import pandas as pd
import argparse
import os

from rule.rule_utils import create_sequential_rule_pairs, calculate_rule_overlap

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
    data_full = pd.read_csv(full_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    data_full.Time = pd.to_datetime(data_full.Time, unit=time_units)
    timeframed = data_full.groupby([pd.Grouper(key=time_key, freq='D')], as_index=False)
    new_rules_ratios = []
    all_rules = set()
    for start_time, data_chunk in timeframed:
        print(start_time)
        rules = create_sequential_rule_pairs(data_chunk, session_key, item_key, time_key)
        overlap_ratio = calculate_rule_overlap(rules, all_rules)
        new_rules_ratios.append(1 - overlap_ratio)
        all_rules.update(rules)
    return pd.DataFrame(new_rules_ratios)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_path', type=str)
    parser.add_argument('--time_units', type=str)
    args = parser.parse_args()
    new_rules_ratio = calculate_new_rules(args.full_path, time_units=args.time_units)
    dataset_name = os.path.split(args.full_path)[-1].split('_')[0]
    new_rules_ratio.to_csv(os.path.join("data", "results", "rule_new", f"{dataset_name}_new_rules.tsv"), sep='\t', index=False, header=False)