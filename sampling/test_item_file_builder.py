import sys
sys.path.append("../GRU4Rec")
import gru4rec
import theano
from theano import tensor as T
import numpy as np, pandas as pd

def create_test_items(gru, train_path, test_path, out_path_prefix, n):
    test = pd.read_csv(test_path, sep='\t', dtype={'ItemId':'str'})
    test_items = test.ItemId.unique()
    test_items = test_items[np.in1d(test_items.astype('int'), gru.itemidmap.index.values.astype('int'))] #Quicker this way, but incorrect if starts with 0
    test_idxs = gru.itemidmap[test_items.astype('str')].values.astype('int32')
    S = (gru.Wy**2).sum(axis=1, keepdims=True)
    WN = theano.shared((gru.Wy / T.sqrt(S)).eval({}))
    I = T.ivector()
    CRAW = WN[I].dot(WN.T)
    CPROC = T.set_subtensor(CRAW[T.arange(I.shape[0]), I], 10)
    ERAW = 2 * T.dot(gru.Wy[I], gru.Wy.T) - S[I] - S.T
    EPROC = T.set_subtensor(ERAW[T.arange(I.shape[0]), I], 10)
    def least_most_sim(scores, n=100):
        indices = np.argsort(scores, axis=1)
        return indices[:, :n], indices[:, -2 : -n - 2 : -1]
    cos = theano.function([I], CPROC)
    euc = theano.function([I], EPROC)

    closest = []
    farthest = []
    similar = []
    dissimilar = []
    for i in range(0, len(test_idxs), 1000):
        scores = cos(test_idxs[i:i+1000])
        least, most = least_most_sim(scores, n=n)
        dissimilar.append(gru.itemidmap.index.values[least.reshape(-1)].reshape(scores.shape[0], -1))
        similar.append(gru.itemidmap.index.values[most.reshape(-1)].reshape(scores.shape[0], -1))
        scores = euc(test_idxs[i:i+1000])
        least, most = least_most_sim(scores, n=n)
        farthest.append(gru.itemidmap.index.values[least.reshape(-1)].reshape(scores.shape[0], -1))
        closest.append(gru.itemidmap.index.values[most.reshape(-1)].reshape(scores.shape[0], -1))
        print('{}/{}'.format(i+len(scores), len(test_idxs)))
        
    closest = np.vstack(closest)
    farthest = np.vstack(farthest)
    similar = np.vstack(similar)
    dissimilar = np.vstack(dissimilar)
    pd.DataFrame(index=test_items, data=closest).to_csv(out_path_prefix + '_closest.tsv', header=None, sep='\t')
    pd.DataFrame(index=test_items, data=farthest).to_csv(out_path_prefix + '_farthest.tsv', header=None, sep='\t')
    pd.DataFrame(index=test_items, data=similar).to_csv(out_path_prefix + '_similar.tsv', header=None, sep='\t')
    pd.DataFrame(index=test_items, data=dissimilar).to_csv(out_path_prefix + '_dissimilar.tsv', header=None, sep='\t')


    uniform = []
    popular = []
    invpopular = []
    popstatic = []
    data = pd.read_csv(train_path, sep='\t', dtype={'ItemId':'str'})
    np.random.seed(42)
    pop = data.groupby('ItemId').size() / len(data)
    toppop = gru.itemidmap[pop.sort_values(ascending=False)[:n+1].index.values].values
    pop = pop[gru.itemidmap.index.values].values
    invpop = 1.0/pop
    invpop = invpop / invpop.sum()
    for i in range(len(test_idxs)):
        inds = np.random.choice(gru.itemidmap.values, size=n+1, replace=False)
        inds = inds[~np.in1d(inds, [test_idxs[i]])][:n]
        uniform.append(gru.itemidmap.index.values[inds])
        
        inds = np.random.choice(gru.itemidmap.values, size=n+1, replace=False, p=pop)
        inds = inds[~np.in1d(inds, [test_idxs[i]])][:n]
        popular.append(gru.itemidmap.index.values[inds])
        
        inds = np.random.choice(gru.itemidmap.values, size=n+1, replace=False, p=invpop)
        inds = inds[~np.in1d(inds, [test_idxs[i]])][:n]
        invpopular.append(gru.itemidmap.index.values[inds])
        
        inds = toppop[~np.in1d(toppop, [test_idxs[i]])][:n]
        popstatic.append(gru.itemidmap.index.values[inds])
        if ((i+1) % 1000 == 0) or ((i+1) == len(test_idxs)):
            print('{}/{}'.format(i+1, len(test_idxs)))
        
    uniform = np.vstack(uniform)
    popular = np.vstack(popular)
    invpopular = np.vstack(invpopular)
    popstatic = np.vstack(popstatic)
    pd.DataFrame(index=test_items, data=uniform).to_csv(out_path_prefix + '_uniform.tsv', header=None, sep='\t')
    pd.DataFrame(index=test_items, data=popular).to_csv(out_path_prefix + '_popular.tsv', header=None, sep='\t')
    pd.DataFrame(index=test_items, data=invpopular).to_csv(out_path_prefix + '_invpopular.tsv', header=None, sep='\t')
    pd.DataFrame(index=test_items, data=popstatic).to_csv(out_path_prefix + '_popstatic.tsv', header=None, sep='\t')

if __name__ == "__main__":
    model_path = '/db_vol/hb_work/rnn/stable/models/coveo_optuna_mrr_bprmax_constrained_fulltrain.pickle'
    train_path = '/db_vol/hb_work/rnn/data/public_raw_data/coveo_ecommerce/coveo_processed_view_train_full.tsv'
    test_path = '/db_vol/hb_work/rnn/data/public_raw_data/coveo_ecommerce/coveo_processed_view_test.tsv'
    out_path_prefix = '/db_vol/hb_work/coveo_test_items_bprmax'
    n = 100
    gru = gru4rec.GRU4Rec.loadmodel(model_path)
    create_test_items(gru=gru, train_path=train_path, test_path=test_path, out_path_prefix=out_path_prefix, n=n)