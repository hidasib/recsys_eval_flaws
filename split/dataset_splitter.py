import numpy as np
import pandas as pd

def create_l1o_split(df, session_key='SessionId', item_key='ItemId', time_key='Time'):
    test_mask = np.concatenate((df.loc[:, session_key].values[:-1] != df.loc[:, session_key].values[1:], [True]))
    session_lengths = df.groupby([session_key]).agg(sl=(item_key,'count'))
    true_test_sessions = session_lengths[session_lengths.sl>2].index.values
    true_test_sessions_mask  = df.loc[:,session_key].isin(true_test_sessions).values
    test_mask = test_mask & true_test_sessions_mask
    train = df[~test_mask].copy()
    session_end_idx = np.nonzero(test_mask)[0]
    test_mask[session_end_idx-1] = True
    test = df[test_mask].copy()
    return train, test

def create_adjusted_time_based_split(df, test_events, session_key='SessionId', item_key='ItemId', time_key='Time'):
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