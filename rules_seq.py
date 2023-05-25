import numpy as np
import pandas as pd
import argparse

def create_sequential_rule_pairs(df, session_key='SessionId', item_key='ItemId', time_key='Time'):
    df_temp = df.copy()
    shifted_itemids = df_temp.loc[:, item_key].iloc[1:].values
    df_temp = df_temp.iloc[:-1]
    df_temp["shifted_item_id"] = shifted_itemids
    session_end_mask = np.concatenate((df_temp.loc[:, session_key].values[:-1] != df_temp.loc[:, session_key].values[1:], [True]))
    df_temp = df_temp[~session_end_mask]
    rules = df_temp.loc[:, item_key] + '_' + df_temp.shifted_item_id
    return rules

def calculate_overlap(rules_test, rules_train):
    rules_test_unique = set(rules_test)
    rules_train_unique = set(rules_train)
    return len(rules_test_unique.intersection(rules_train_unique)) / len(rules_test_unique)

def create_l1o_split(df, session_key='SessionId', item_key='ItemId', time_key='Time'):
    test_mask = np.concatenate((df.loc[:, session_key].values[:-1] != df.loc[:, session_key].values[1:], [True]))
    session_lengths = df.groupby([session_key]).agg(sl=(item_key,'count'))
    true_test_sessions = session_lengths[session_lengths.sl>2].index.values
    true_test_sessions_mask  = df.loc[:,session_key].isin(true_test_sessions).values
    test_mask = test_mask & true_test_sessions_mask
    df_train = df[~test_mask].copy()
    session_end_idx = np.nonzero(test_mask)[0]
    test_mask[session_end_idx-1] = True
    df_test = df[test_mask].copy()
    return df_train, df_test

def create_time_based_split(df, test_events, session_key='SessionId', item_key='ItemId', time_key='Time'):
    sbeg_slen = df.groupby([session_key], as_index=False).agg(sbeg=(time_key,"min"), slen=(session_key, 'count'))
    sbeg_slen = sbeg_slen.sort_values(by=["sbeg"], ascending=False)
    sbeg_slen["total_events"] = np.cumsum(sbeg_slen.slen.values - 1) # -1 for start item
    test_ids_begs = sbeg_slen[sbeg_slen.total_events <= test_events][[session_key, "sbeg"]]
    test_sess_ids, test_sess_begs = test_ids_begs.loc[:, session_key], test_ids_begs.sbeg
    tsplit_test = test_sess_begs.min()
    test = df[df.loc[:,session_key].isin(test_sess_ids)]
    train = df[df.loc[:,time_key] < tsplit_test]
    session_length = train.groupby(session_key).size()
    train = train[train.loc[:,session_key].isin(session_length[session_length > 1].index)]
    return train, test

def rule_overlap(full_path, train_path, test_path, session_key='SessionId', item_key='ItemId', time_key='Time'):
    data_full = pd.read_csv(full_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    train_l1o, test_l1o = create_l1o_split(data_full)
    print("Calculating overlap for 'leave-one-out' technique")
    train_rules_l1o = create_sequential_rule_pairs(train_l1o)
    test_rules_l1o = create_sequential_rule_pairs(test_l1o)
    overlap_l1o = calculate_overlap(test_rules_l1o, train_rules_l1o)
    train_events_l1o = len(train_l1o)
    test_events_l1o = len(test_l1o)-test_l1o.loc[:,session_key].nunique()
    print(f"\ttrain_ratio: {train_events_l1o/len(data_full):.4f}", f"test_ratio: {test_events_l1o/len(data_full):.4f}")
    print(f"\toverlap full l1o: {overlap_l1o:.8f}")

    print(f"Calculating overlap for 'timebased split' technique, with test size adjusted to l1o test size {test_events_l1o/len(data_full):.4f}")
    train_tb, test_tb = create_time_based_split(data_full, test_events=test_events_l1o)
    train_rules_tb = create_sequential_rule_pairs(train_tb)
    test_rules_tb = create_sequential_rule_pairs(test_tb)
    overlap_tb = calculate_overlap(test_rules_tb, train_rules_tb)
    train_events_tb = len(train_tb)
    test_events_tb = len(test_tb)-test_tb.loc[:,session_key].nunique()
    print(f"\ttrain_ratio: {train_events_tb/len(data_full):.4f}", f"test_ratio: {test_events_tb/len(data_full):.4f}")
    print(f"\toverlap full l1o: {overlap_tb:.8f}")

    # train_tb = pd.read_csv(train_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    # test_tb = pd.read_csv(test_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    # train_rules_tb = create_sequential_rule_pairs(train_tb)
    # test_rules_tb = create_sequential_rule_pairs(test_tb)
    # overlap_tb = calculate_overlap(test_rules_tb, train_rules_tb)
    # print(f"overlap timebased original: {overlap_tb:.8f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_path', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    args = parser.parse_args()

    rule_overlap(args.full_path, args.train_path, args.test_path)