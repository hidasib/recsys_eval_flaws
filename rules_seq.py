import numpy as np
import pandas as pd
import argparse

from split.dataset_splitter import create_l1o_split, create_adjusted_time_based_split
from split.rules import create_sequential_rule_pairs, calculate_rule_overlap

def rule_overlap(full_path, train_path, test_path, session_key='SessionId', item_key='ItemId', time_key='Time'):
    data_full = pd.read_csv(full_path, sep='\t', dtype={session_key:str, item_key:str, time_key:int})
    train_l1o, test_l1o = create_l1o_split(data_full)
    print("Calculating overlap for 'leave-one-out' technique")
    train_rules_l1o = create_sequential_rule_pairs(train_l1o)
    test_rules_l1o = create_sequential_rule_pairs(test_l1o)
    overlap_l1o = calculate_rule_overlap(test_rules_l1o, train_rules_l1o)
    train_events_l1o = len(train_l1o)
    test_events_l1o = len(test_l1o)-test_l1o.loc[:,session_key].nunique()

    print(f"\ttrain shape: {train_l1o.shape}", f"test shape: {test_l1o.shape}", f"test events: {test_events_l1o}")
    print(f"\ttrain_ratio: {train_events_l1o/len(data_full):.4f}", f"test_ratio: {test_events_l1o/len(data_full):.4f}")
    print(f"\toverlap full l1o: {overlap_l1o:.8f}")

    print(f"Calculating overlap for 'timebased split' technique, with test size adjusted to l1o test size {test_events_l1o/len(data_full):.4f}")
    train_tb, test_tb = create_adjusted_time_based_split(data_full, test_events=test_events_l1o)
    train_rules_tb = create_sequential_rule_pairs(train_tb)
    test_rules_tb = create_sequential_rule_pairs(test_tb)
    overlap_tb = calculate_rule_overlap(test_rules_tb, train_rules_tb)
    train_events_tb = len(train_tb)
    test_events_tb = len(test_tb)-test_tb.loc[:,session_key].nunique()

    print(f"\ttrain shape: {train_tb.shape}", f"\ttest shape: {test_tb.shape}", f"test events: {test_events_tb}")
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