import numpy as np
import pandas as pd

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