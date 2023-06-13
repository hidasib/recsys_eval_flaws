import numpy as np
import pandas as pd
import gc
import datetime as dt
import glob
import argparse
import os

def main(path: str):
    data = pd.read_csv(os.path.join(path,"ratings.dat"), sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], dtype={'user_id':str, 'item_id':str, 'rating':str, 'timestamp':int})
 
    data = data.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    data["session_id"] = np.cumsum(
        np.hstack(
            [
                False,
                (data.user_id.values[1:] != data.user_id.values[:-1]),
            ]
        )
    )

    view = pd.DataFrame(
        {
            "session_id": data.session_id.values,
            "item_id": data.item_id.values,
            "time": data.timestamp.values,
        }
    )

    dupmask = np.hstack(
        [
            False,
            (view.session_id.values[1:] == view.session_id.values[:-1])
            & (view.item_id.values[1:] == view.item_id.values[:-1]),
        ]
    )
    view_dedup = view[~dupmask].copy()

    pl = -1
    print(
        len(view_dedup),
        view_dedup.session_id.nunique(),
        view_dedup.item_id.nunique(),
    )
    while len(view_dedup) != pl:
        pl = len(view_dedup)
        slength = view_dedup.groupby("session_id").size()
        view_dedup.drop(
            view_dedup[view_dedup.session_id.isin(slength[slength == 1].index)].index,
            inplace=True,
        )
        gc.collect()
        print(
            len(view_dedup),
            view_dedup.session_id.nunique(),
            view_dedup.item_id.nunique(),
        )
        isupp = view_dedup.groupby("item_id").size()
        view_dedup.drop(
            view_dedup[view_dedup.item_id.isin(isupp[isupp < 5].index)].index, inplace=True
        )
        gc.collect()
        print(
            len(view_dedup),
            view_dedup.session_id.nunique(),
            view_dedup.item_id.nunique(),
        )

    processed = pd.DataFrame(
        {
            "SessionId": view_dedup.session_id.values,
            "ItemId": view_dedup.item_id.values,
            "Time": view_dedup.time.values,
        }
    )
    processed.to_csv(
        f"{path}/ml10m_processed_view_full.tsv",
        sep="\t",
        index=False,
    )

    sbeg = processed.groupby("SessionId").Time.min()
    tsplit = sbeg.max()
    tday = 86400
    days = 8 * 7
    test = processed[processed.SessionId.isin(sbeg[sbeg >= tsplit - tday * days].index)]
    train = processed[processed.Time < tsplit - tday * days]
    test.to_csv(
        f"{path}/ml10m_processed_view_test.tsv",
        sep="\t",
        index=False,
    )
    session_length = train.groupby("SessionId").size()
    train = train[train.SessionId.isin(session_length[session_length > 1].index)]
    train.to_csv(
        f"{path}/ml10m_processed_view_train_full.tsv",
        sep="\t",
        index=False,
    )

    train2 = train[train.Time < tsplit - tday * days * 2]
    test2 = train[train.SessionId.isin(sbeg[sbeg >= tsplit - tday * days * 2].index)]
    session_length2 = train2.groupby("SessionId").size()
    train2 = train2[train2.SessionId.isin(session_length2[session_length2 > 1].index)]
    train2.to_csv(
        f"{path}/ml10m_processed_view_train_tr.tsv",
        sep="\t",
        index=False,
    )
    test2.to_csv(
        f"{path}/ml10m_processed_view_train_valid.tsv",
        sep="\t",
        index=False,
    )

    names = []
    num_events = []
    num_sessions = []
    num_items = []
    num_days = []
    start_times = []
    end_times = []
    length_min = []
    length_max = []
    length_avg = []
    item_view_avg = []
    diff_min = []
    diff_max = []
    for curr_path in glob.glob(
        f"{path}/ml10m_processed_view*"
    ):
        data = pd.read_csv(curr_path, sep="\t", dtype={"ItemId": "str"})
        data.Time = data.Time
        names.append(curr_path.split("/")[-1])
        num_events.append(len(data))
        num_sessions.append(data.SessionId.nunique())
        num_items.append(data.ItemId.nunique())
        num_days.append((data.Time.max() - data.Time.min()) / tday)
        start_times.append(
            dt.datetime.utcfromtimestamp(data.Time.min()).strftime("%Y-%m-%d %H:%M:%S.%f")
        )
        end_times.append(
            dt.datetime.utcfromtimestamp(data.Time.max()).strftime("%Y-%m-%d %H:%M:%S.%f")
        )
        slength = data.groupby("SessionId").size()
        itemview = data.groupby("ItemId").size()
        sdiff = data.groupby("SessionId").Time.max() - data.groupby("SessionId").Time.min()
        sdiff = sdiff
        length_min.append(slength.min())
        length_max.append(slength.max())
        length_avg.append(slength.mean())
        item_view_avg.append(itemview.mean())
        diff_min.append(sdiff.min())
        diff_max.append(sdiff.max())
    stats = pd.DataFrame(
        {
            "Dataset": names,
            "NumEvents": num_events,
            "NumSessions": num_sessions,
            "NumItems": num_items,
            "NumDays": num_days,
            "StartTime": start_times,
            "EndTime": end_times,
            "AvgItemViews": item_view_avg,
            "MinSessionLength": length_min,
            "MaxSessionLength": length_max,
            "AvgSessionLength": length_avg,
            "MinSessionTime (sec)": diff_min,
            "MaxSessionTime (sec)": diff_max,
        }
    )
    print(stats)
    stats.to_csv(
        f"{path}/stats.tsv",
        sep="\t",
        index=False,
        float_format='%.4f',
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="Path of the directory of the dataset",
        required=True,
    )
    args = parser.parse_args()
    main(path=args.path)