import numpy as np
import pandas as pd
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def get_rank_sampling(gru, test_data, negative_items_file, session_key='SessionId', item_key='ItemId', time_key='Time', batch_size=100, mode='standard'):
    if gru.error_during_train: raise Exception
    use_context = False
    if hasattr(gru, 'context_cols'):
        use_context = True
        context_cols = gru.context_cols
    srng = RandomStreams()
    I = T.imatrix()
    X = T.ivector()
    Y = T.ivector()
    M = T.iscalar()
    yhat, H, updatesH = gru.symbolic_predict(X, Y, M, None, batch_size)
    if mode == 'tiebreaking': yhat += srng.uniform(size=yhat.shape) * 1e-10  
    yhat = yhat[T.arange(M).dimshuffle((0, 'x')), I].T
    targets = yhat[0]
    others = yhat   
    if mode == 'standard': ranks = (others > targets).sum(axis=0) + 1
    elif mode == 'conservative': ranks = (others >= targets).sum(axis=0)
    elif mode == 'median':  ranks = (others > targets).sum(axis=0) + 0.5*((others == targets).sum(axis=0) - 1) + 1
    elif mode == 'tiebreaking': ranks = (others > targets).sum(axis=0) + 1
    else: raise NotImplementedError
    evaluate = theano.function(inputs=[X, Y, M, I], outputs=ranks, updates=updatesH, allow_input_downcast=True, on_unused_input='ignore')
    test_data = pd.merge(test_data, pd.DataFrame({'ItemIdx':gru.itemidmap.values, gru.item_key:gru.itemidmap.index}), on=gru.item_key, how='inner')
    test_data.sort_values([gru.session_key, gru.time_key, gru.item_key], inplace=True)
    test_data_items = test_data.ItemIdx.values
    iters = np.arange(batch_size)
    maxiter = iters.max()
    offset_sessions = np.zeros(test_data[gru.session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(gru.session_key).size().cumsum()
    start = offset_sessions[iters]
    end = offset_sessions[iters+1]
    #TODO if prod: remove unknown items and duplications
    neg_data = pd.read_csv(negative_items_file, sep='\t', header=None, dtype='str')
    neg_data = gru.itemidmap[neg_data.values.reshape(-1)].values.reshape(len(neg_data), -1)
    neg_samples = dict()
    for i in range(len(neg_data)):
        neg_samples[neg_data[i][0]] = neg_data[i][1:]
    finished = False
    res = []
    while not finished:
        minlen = (end-start).min()
        out_idx = test_data_items[start]
        for i in range(minlen-1):
            in_idx = out_idx
            out_idx = test_data_items[start+i+1]
            y = out_idx
            negs = np.hstack([out_idx.reshape(-1, 1), np.array([neg_samples[oidx] for oidx in out_idx], dtype='int64')])
            res.append(evaluate(in_idx, y, len(iters), negs))
        start = start+minlen-1
        finished_mask = (end-start<=1)
        n_finished = finished_mask.sum()
        iters[finished_mask] = maxiter + np.arange(1,n_finished+1)
        maxiter += n_finished
        valid_mask = (iters < len(offset_sessions)-1)
        n_valid = valid_mask.sum()
        if n_valid == 0:
            finished = True
            break
        mask = finished_mask & valid_mask
        sessions = iters[mask]
        start[mask] = offset_sessions[sessions]
        end[mask] = offset_sessions[sessions+1]
        iters = iters[valid_mask]
        start = start[valid_mask]
        end = end[valid_mask]
        if valid_mask.any():
            for i in range(len(H)):
                tmp = H[i].get_value(borrow=True)
                tmp[mask] = 0
                tmp = tmp[valid_mask]
                H[i].set_value(tmp, borrow=True)
    return np.hstack(res)

def get_rank(gru, test_data, session_key='SessionId', item_key='ItemId', time_key='Time', batch_size=100, mode='standard'):
    if gru.error_during_train: raise Exception
    srng = RandomStreams()
    X = T.ivector()
    Y = T.ivector()
    M = T.iscalar()
    yhat, H, updatesH = gru.symbolic_predict(X, Y, M, None, batch_size)
    if mode == 'tiebreaking': yhat += srng.uniform(size=yhat.shape) * 1e-10
    targets = T.diag(yhat.T[Y])
    others = yhat.T
    if mode == 'standard': ranks = (others > targets).sum(axis=0) + 1
    elif mode == 'conservative': ranks = (others >= targets).sum(axis=0)
    elif mode == 'median':  ranks = (others > targets).sum(axis=0) + 0.5*((others == targets).sum(axis=0) - 1) + 1
    elif mode == 'tiebreaking': ranks = (others > targets).sum(axis=0) + 1
    else: raise NotImplementedError
    evaluate = theano.function(inputs=[X, Y, M], outputs=ranks, updates=updatesH, allow_input_downcast=True, on_unused_input='ignore')
    test_data = pd.merge(test_data, pd.DataFrame({'ItemIdx':gru.itemidmap.values, gru.item_key:gru.itemidmap.index}), on=gru.item_key, how='inner')
    test_data.sort_values([gru.session_key, gru.time_key, gru.item_key], inplace=True)
    test_data_items = test_data.ItemIdx.values
    iters = np.arange(batch_size)
    maxiter = iters.max()
    offset_sessions = np.zeros(test_data[gru.session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(gru.session_key).size().cumsum()
    start = offset_sessions[iters]
    end = offset_sessions[iters+1]
    finished = False
    res = []
    while not finished:
        minlen = (end-start).min()
        out_idx = test_data_items[start]
        for i in range(minlen-1):
            in_idx = out_idx
            out_idx = test_data_items[start+i+1]
            y = out_idx
            res.append(evaluate(in_idx, y, len(iters)))
        start = start+minlen-1
        finished_mask = (end-start<=1)
        n_finished = finished_mask.sum()
        iters[finished_mask] = maxiter + np.arange(1,n_finished+1)
        maxiter += n_finished
        valid_mask = (iters < len(offset_sessions)-1)
        n_valid = valid_mask.sum()
        if n_valid == 0:
            finished = True
            break
        mask = finished_mask & valid_mask
        sessions = iters[mask]
        start[mask] = offset_sessions[sessions]
        end[mask] = offset_sessions[sessions+1]
        iters = iters[valid_mask]
        start = start[valid_mask]
        end = end[valid_mask]
        if valid_mask.any():
            for i in range(len(H)):
                tmp = H[i].get_value(borrow=True)
                tmp[mask] = 0
                tmp = tmp[valid_mask]
                H[i].set_value(tmp, borrow=True)
    return np.hstack(res)

def get_rank_uniform(gru, test_data, nsample, session_key='SessionId', item_key='ItemId', time_key='Time', batch_size=100, mode='standard'):
    if gru.error_during_train: raise Exception
    use_context = False
    if hasattr(gru, 'context_cols'):
        use_context = True
        context_cols = gru.context_cols
    srng = RandomStreams()
    I = T.imatrix()
    X = T.ivector()
    Y = T.ivector()
    M = T.iscalar()
    yhat, H, updatesH = gru.symbolic_predict(X, Y, M, None, batch_size)
    if mode == 'tiebreaking': yhat += srng.uniform(size=yhat.shape) * 1e-10  
    yhat = yhat[T.arange(M).dimshuffle((0, 'x')), I].T
    targets = yhat[0]
    others = yhat   
    if mode == 'standard': ranks = (others > targets).sum(axis=0) + 1
    elif mode == 'conservative': ranks = (others >= targets).sum(axis=0)
    elif mode == 'median':  ranks = (others > targets).sum(axis=0) + 0.5*((others == targets).sum(axis=0) - 1) + 1
    elif mode == 'tiebreaking': ranks = (others > targets).sum(axis=0) + 1
    else: raise NotImplementedError
    evaluate = theano.function(inputs=[X, Y, M, I], outputs=ranks, updates=updatesH, allow_input_downcast=True, on_unused_input='ignore')
    test_data = pd.merge(test_data, pd.DataFrame({'ItemIdx':gru.itemidmap.values, gru.item_key:gru.itemidmap.index}), on=gru.item_key, how='inner')
    test_data.sort_values([gru.session_key, gru.time_key, gru.item_key], inplace=True)
    test_data_items = test_data.ItemIdx.values
    iters = np.arange(batch_size)
    maxiter = iters.max()
    offset_sessions = np.zeros(test_data[gru.session_key].nunique()+1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(gru.session_key).size().cumsum()
    start = offset_sessions[iters]
    end = offset_sessions[iters+1]
    #TODO if prod: remove unknown items and duplications
    ones = np.ones(len(gru.itemidmap))
    finished = False
    res = []
    while not finished:
        minlen = (end-start).min()
        out_idx = test_data_items[start]
        for i in range(minlen-1):
            in_idx = out_idx
            out_idx = test_data_items[start+i+1]
            y = out_idx
            negs = []
            for oidx in out_idx:
                neg = np.zeros(nsample+1, dtype='int64')
                neg[0] = oidx
                ones[oidx] = 0
                distr = np.cumsum(ones) / (len(ones) - 1)
                distr[-1] = 1
                neg[1:] = np.searchsorted(distr, np.random.rand(nsample))
                negs.append(neg)
                ones[oidx] = 1
            negs = np.vstack(negs)
            #negs = np.hstack([out_idx.reshape(-1, 1), np.array([neg_samples[oidx] for oidx in out_idx], dtype='int64')])
            res.append(evaluate(in_idx, y, len(iters), negs))
        start = start+minlen-1
        finished_mask = (end-start<=1)
        n_finished = finished_mask.sum()
        iters[finished_mask] = maxiter + np.arange(1,n_finished+1)
        maxiter += n_finished
        valid_mask = (iters < len(offset_sessions)-1)
        n_valid = valid_mask.sum()
        if n_valid == 0:
            finished = True
            break
        mask = finished_mask & valid_mask
        sessions = iters[mask]
        start[mask] = offset_sessions[sessions]
        end[mask] = offset_sessions[sessions+1]
        iters = iters[valid_mask]
        start = start[valid_mask]
        end = end[valid_mask]
        if valid_mask.any():
            for i in range(len(H)):
                tmp = H[i].get_value(borrow=True)
                tmp[mask] = 0
                tmp = tmp[valid_mask]
                H[i].set_value(tmp, borrow=True)
    return np.hstack(res)