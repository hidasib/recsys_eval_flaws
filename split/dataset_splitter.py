import numpy as np
import pandas as pd
from typing import Tuple

def create_l1o_split(df: pd.DataFrame, force_test:bool=False, duplicate_penultimate_event:bool=False, n_session_sample:int=None, session_key:str='SessionId', item_key:str='ItemId', time_key:int='Time') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a dataframe to train and test sets using leave-one-out technique.

    Args:
        df (pd.DataFrame): The data to be split
        force_test (bool, optional): Whether to force session with length 2 to the test set. If False, only sessions with at least 3 events will be considered when selectig the test set. Defaults to False.
        duplicate_penultimate_event (bool, optional): If True, in each session, the penultimate event will be present in both the train and test set. Defaults to False.
        n_session_sample (int, optional): When provided, the test set will contain n_session_sample randomly sampled sessions. Defaults to None.
        session_key (str, optional): Column name of the sessin ID. Defaults to 'SessionId'.
        item_key (str, optional): Column name of the item ID. Defaults to 'ItemId'.
        time_key (int, optional): Column name of the timestamp. Defaults to 'Time'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test data
    """
    test_mask = np.concatenate((df.loc[:, session_key].values[:-1] != df.loc[:, session_key].values[1:], [True]))
    session_lengths = df.groupby([session_key]).agg(sl=(item_key,'count'))
    if not force_test:
        true_test_sessions = session_lengths[session_lengths.sl>2].index.values
    else:
        true_test_sessions = session_lengths.index.values
    if n_session_sample is not None:
        np.random.seed(42)
        true_test_sessions = np.random.choice(true_test_sessions, size=n_session_sample, replace=False)
    true_test_sessions_mask  = df.loc[:,session_key].isin(true_test_sessions).values
    test_mask = test_mask & true_test_sessions_mask
    train = df[~test_mask]
    session_end_idx = np.nonzero(test_mask)[0]
    if duplicate_penultimate_event:
        test_mask[session_end_idx-1] = True
    test = df[test_mask]
    return train, test

def create_adjusted_time_based_split(df:pd.DataFrame, test_events:int, session_key:str='SessionId', item_key:str='ItemId', time_key:str='Time') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a dataframe to train and test sets, with test events matching the test_events parameter, using timebased split technique.

    Args:
        df (pd.DataFrame): The data to be split
        test_events (int): The number of events to be scored (:discounting the starting event of each session) in the test set   
        session_key (str, optional): Column name of the sessin ID. Defaults to 'SessionId'.
        item_key (str, optional): Column name of the item ID. Defaults to 'ItemId'.
        time_key (int, optional): Column name of the timestamp. Defaults to 'Time'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test data
    """
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