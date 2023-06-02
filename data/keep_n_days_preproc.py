import pandas as pd 
import argparse
import os
from typing import List

def keep_last_n_days(train_path:str, ndays:List[int], test_path:str=None, session_key:str="SessionId", item_key:str="ItemId", time_key:str="Time"):
    data = pd.read_csv(train_path, sep="\t", dtype={item_key:str, time_key:int})
    train_dir_path, train_file_name = os.path.split(train_path)
    if test_path is not None:
        test_rescale = pd.DataFrame({"dataset":[], "test_new_targets":[], "test_orig_targets":[], "scale":[]})
        test = pd.read_csv(test_path, sep="\t", dtype={item_key:str, time_key:int})
        test_orig_targets = len(test) - test[session_key].nunique()
        test_rescale.loc[0] = [train_file_name, test_orig_targets, test_orig_targets, test_orig_targets/test_orig_targets]
    for i, nday in enumerate(ndays):
        tdata = data.drop(data[data.Time < data.Time.max() - 86400000 * nday].index)
        print(len(tdata), tdata.SessionId.nunique(), tdata.ItemId.nunique())
        new_file_name = train_file_name[:-4] + f"_{nday}D" + train_file_name[-4:]
        tdata.to_csv(os.path.join(train_dir_path, new_file_name), sep="\t", index=False)
        if test_path is not None:
            test_new_targets = test.ItemId.isin(tdata.ItemId.unique()).sum() - test[session_key].nunique()
            test_rescale.loc[i+1] = [new_file_name, test_new_targets, test_orig_targets, test_new_targets/test_orig_targets]
    if test_path is not None:
        print(test_rescale)
        test_rescale.to_csv(os.path.join(train_dir_path, f"{train_file_name.split('_')[0]}_recall_rescale.tsv"), sep='\t', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="Path to the train dataset", required=True)
    parser.add_argument("--train_tr_path", type=str, help="Path to the train_tr dataset", required=True)
    parser.add_argument("--test_path", type=str, help="Path to the test dataset", required=True)
    parser.add_argument("-n", "--ndays", type=int, nargs='+', default=[91,56,28,14,7])
    parser.add_argument('--session_key', type=str, default="SessionId")
    parser.add_argument('--item_key', type=str, default="ItemId")
    parser.add_argument('--time_key', type=str, default="Time")
    args = parser.parse_args()

    print("Split for train")
    keep_last_n_days(train_path=args.train_path, test_path=args.test_path, ndays=args.ndays)
    print("Split for train tr")
    keep_last_n_days(train_path=args.train_tr_path, ndays=args.ndays)



