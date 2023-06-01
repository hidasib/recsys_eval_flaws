import pandas as pd 
import argparse
import os
from typing import List

def keep_last_n_days(train_path:str, ndays:List[int], session_key:str="SessionId", item_key:str="ItemId", time_key:str="Time"):
    data = pd.read_csv(train_path, sep="\t", dtype={item_key:str, time_key:int})
    train_dir_path, train_file_name = os.path.split(train_path)
    for nday in ndays:
        tdata = data.drop(data[data.Time < data.Time.max() - 86400000 * nday].index)
        print(len(tdata), tdata.SessionId.nunique(), tdata.ItemId.nunique())
        new_file_name = train_file_name[:-4] + f"_{nday}D" + train_file_name[-4:]
        tdata.to_csv(os.path.join(train_dir_path, new_file_name), sep="\t", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, help="Path to the train dataset", required=True)
    parser.add_argument("--train_tr_path", type=str, help="Path to the train_tr dataset", required=True)
    parser.add_argument("-n", "--ndays", type=int, nargs='+', default=[91,56,28,14,7])
    parser.add_argument('--session_key', type=str, default="SessionId")
    parser.add_argument('--item_key', type=str, default="ItemId")
    parser.add_argument('--time_key', type=str, default="Time")
    args = parser.parse_args()

    keep_last_n_days(train_path=args.train_path, ndays=args.ndays)
    keep_last_n_days(train_path=args.train_tr_path, ndays=args.ndays)



