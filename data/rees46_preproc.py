import numpy as np
import pandas as pd
import gc
import datetime as dt
import glob
import argparse
import os


def main(path: str):
    data = pd.read_csv(
        os.path.join(path, "2019-Nov.csv"),
        sep=",",
        usecols=["event_time", "event_type", "product_id", "user_id"],
        dtype={
            "event_type": "category",
            "product_id": "str",
            "user_id": "str",
        },
        parse_dates=["event_time"],
    )
    data2 = pd.read_csv(
        os.path.join(path, "2019-Oct.csv"),
        sep=",",
        usecols=["event_time", "event_type", "product_id", "user_id"],
        dtype={
            "event_type": "category",
            "product_id": "str",
            "user_id": "str",
        },
        parse_dates=["event_time"],
    )
    data = pd.concat([data2, data])
    users = data.user_id.unique()
    items = data.product_id.unique()
    data = pd.merge(
        data,
        pd.DataFrame({"user_id": users, "useridx": np.arange(len(users))}),
        on="user_id",
    )
    data = pd.merge(
        data,
        pd.DataFrame({"product_id": items, "itemidx": np.arange(len(items))}),
        on="product_id",
    )
    data["time"] = data.event_time.view("int") // 10**9
    
    data.sort_values(["useridx", "time", "itemidx"], inplace=True)
    # view = data[data.event_type == "view"]
    view = data.drop(data[data.event_type != "view"].index)

    view["sessionidx"] = np.cumsum(
        np.hstack(
            [
                False,
                (view.time.values[1:] - view.time.values[:-1] > 3600)
                | (view.useridx.values[1:] != view.useridx.values[:-1]),
            ]
        )
    )

    dupmask = np.hstack(
        [
            False,
            (view.sessionidx.values[1:] == view.sessionidx.values[:-1])
            & (view.itemidx.values[1:] == view.itemidx.values[:-1]),
        ]
    )
    view_dedup = view[~dupmask]
    
    pl = -1
    print(
        len(view_dedup),
        view_dedup.useridx.nunique(),
        view_dedup.sessionidx.nunique(),
        view_dedup.itemidx.nunique(),
    )
    while len(view_dedup) != pl:
        pl = len(view_dedup)
        slength = view_dedup.groupby("sessionidx").size()
        view_dedup.drop(
            view_dedup[view_dedup.sessionidx.isin(slength[slength == 1].index)].index,
            inplace=True,
        )
        gc.collect()
        print(
            len(view_dedup),
            view_dedup.useridx.nunique(),
            view_dedup.sessionidx.nunique(),
            view_dedup.itemidx.nunique(),
        )
        isupp = view_dedup.groupby("itemidx").size()
        view_dedup.drop(
            view_dedup[view_dedup.itemidx.isin(isupp[isupp < 5].index)].index,
            inplace=True,
        )
        gc.collect()
        print(
            len(view_dedup),
            view_dedup.useridx.nunique(),
            view_dedup.sessionidx.nunique(),
            view_dedup.itemidx.nunique(),
        )
    
    processed = pd.DataFrame(
        {
            "SessionId": view_dedup.sessionidx.values,
            "ItemId": view_dedup.product_id.values,
            "Time": view_dedup.time.values,
        }
    )
    processed.to_csv(
        os.path.join(path, "rees46_processed_view_userbased_full.tsv"),
        sep="\t",
        index=False,
    )

    sbeg = processed.groupby("SessionId").Time.min()
    tsplit = 1575158400
    tday = 86400
    test = processed[processed.SessionId.isin(sbeg[sbeg >= tsplit - tday].index)]
    train = processed[processed.Time < tsplit - tday]
    test.to_csv(
        os.path.join(path, "rees46_processed_view_userbased_test.tsv"),
        sep="\t",
        index=False,
    )
    sslength = train.groupby("SessionId").size()
    train = train[train.SessionId.isin(sslength[sslength > 1].index)]
    train.to_csv(
        os.path.join(path, "rees46_processed_view_userbased_train_full.tsv"),
        sep="\t",
        index=False,
    )

    train2 = train[train.Time < tsplit - tday * 2]
    test2 = train[train.SessionId.isin(sbeg[sbeg >= tsplit - tday * 2].index)]
    sslength2 = train2.groupby("SessionId").size()
    train2 = train2[train2.SessionId.isin(sslength2[sslength2 > 1].index)]
    train2.to_csv(
        os.path.join(path, "rees46_processed_view_userbased_train_tr.tsv"),
        sep="\t",
        index=False,
    )
    test2.to_csv(
        os.path.join(path, "rees46_processed_view_userbased_train_valid.tsv"),
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
    for curr_path in glob.glob(f"{path}/rees46_processed_view*"):
        data = pd.read_csv(curr_path, sep="\t", dtype={"ItemId": "str"})
        data.Time = data.Time
        names.append(curr_path.split("/")[-1])
        num_events.append(len(data))
        num_sessions.append(data.SessionId.nunique())
        num_items.append(data.ItemId.nunique())
        num_days.append((data.Time.max() - data.Time.min()) / tday)
        start_times.append(
            dt.datetime.utcfromtimestamp(data.Time.min()).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
        )
        end_times.append(
            dt.datetime.utcfromtimestamp(data.Time.max()).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
        )
        slength = data.groupby("SessionId").size()
        itemview = data.groupby("ItemId").size()
        sdiff = (
            data.groupby("SessionId").Time.max() - data.groupby("SessionId").Time.min()
        )
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
        float_format="%.4f",
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
