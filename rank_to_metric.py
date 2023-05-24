import sys
sys.path.append("./GRU4Rec")
import numpy as np, pandas as pd
import gru4rec
from sampling import evaluation_utils, test_item_file_builder
import os
import joblib
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_file_path', type=str)
	parser.add_argument('--train_file_path', type=str)
	parser.add_argument('--model_path', type=str)
	args = parser.parse_args()

	test_file_path = args.test_file_path
	model_name = os.path.split(args.model_path)[-1][:-7]

	test = pd.read_csv(test_file_path, sep='\t', dtype={'ItemId':'str'})
	test_path, test_file = os.path.split(test_file_path)
	gruA = gru4rec.GRU4Rec.loadmodel(args.model_path)
	s_random_names = ['full', '0.1'] #, '0.01', '0.001', '100']
	s_sim_names = ["closest", "farthest" ] #, "similar", "dissimilar", "uniform", "popular", "invpopular", "popstatic"]
	for sname in  s_random_names + s_sim_names:
		if not os.path.isfile(os.path.join('data', 'results', model_name + '_' + sname + '_rank.pickle')):
			print('Building rank file for {} for {} samples'.format(model_name, sname))
			if sname == 'full':
				ranksA = evaluation_utils.get_rank(gruA, test, mode='conservative')
			elif sname in s_random_names:
				if sname == '100':
					nsample = 100
				else:
					nsample = int(float(sname) * len(gruA.itemidmap))
				np.random.seed(42)
				ranksA = evaluation_utils.get_rank_uniform(gruA, test, nsample, mode='conservative')
			else:
				nsample = 100
				negative_items_file = model_name + '_' + sname
				if not os.path.isfile(os.path.join(test_path, negative_items_file + ".tsv")):
					print(f"Building negative item files for: {s_sim_names}")
					test_item_file_builder.create_test_items(gruA, n=nsample, train_path=args.train_file_path, test_path=test_file_path, out_path_prefix=os.path.join(test_path, model_name))
				_rec, _mrr = evaluation_utils.evaluate_sampling(gruA, test, mode='conservative', negative_items_file=os.path.join(test_path, negative_items_file + ".tsv"))
				print(_rec, _mrr)
				ranksA = evaluation_utils.evaluate_sampling2(gruA, test, mode='conservative', negative_items_file=os.path.join(test_path, negative_items_file + ".tsv"))
			joblib.dump(ranksA, os.path.join('data', 'results', model_name + '_' + sname + '_rank.pickle'))
		else:
			ranksA = joblib.load(os.path.join('data', 'results', model_name + '_' + sname + '_rank.pickle'))
		print('Computing recall/mrr for {} for {} samples'.format(model_name, sname))
		print(np.min(ranksA), np.max(ranksA))
		ranks_df = pd.DataFrame({"ranks": ranksA.flatten()}).groupby("ranks", as_index=False).agg(sizes=('ranks', 'count'))
		ranks_df.sort_values(by=["ranks"],inplace=True)
		print(ranks_df)
		if sname == 'full':
			nsample = len(gruA.itemidmap)
		elif (sname == '100') or (sname in s_sim_names):
			nsample = 100
		else:
			nsample = int(float(sname) * len(gruA.itemidmap))
		n = nsample + 1
		measures = ranks_df.sizes.sum()
		recall = np.zeros(n)
		recall[ranks_df.ranks.values-1] = ranks_df.sizes.values
		recall = np.cumsum(recall) / measures
		mrr = np.zeros(n)
		mrr[ranks_df.ranks.values-1] = ranks_df.sizes.values / ranks_df.ranks.values
		mrr = np.cumsum(mrr) / measures
		print(recall[0], recall[4],recall[9],recall[19])
		print(mrr[0], mrr[4],mrr[9],mrr[19])
		pd.DataFrame(data=recall).to_csv(os.path.join('data', 'results', model_name + '_' + sname + '_recall.tsv'), index=False, header=None)
		pd.DataFrame(data=mrr).to_csv(os.path.join('data', 'results', model_name + '_' + sname + '_mrr.tsv'), index=False, header=None)


		# rec, mrr = [], []
		# #TODO: speedup
		# if sname == 'full':
		# 	nsample = len(gruA.itemidmap)
		# for i in range(1, nsample + 1, 1000):
		# 	mask = (np.arange(i, min(i + 1000, nsample + 1), 1).reshape(-1, 1) >= ranksA)
		# 	rec.append(mask.mean(axis=1))
		# 	mrr.append((mask / ranksA).mean(axis=1))
		# rec = np.hstack(rec)
		# mrr = np.hstack(mrr)
		# pd.DataFrame(data=rec.reshape(-1,1)).to_csv(os.path.join('results', model_name + sname + '_recall.tsv'), index=False, header=None)
		# pd.DataFrame(data=mrr.reshape(-1,1)).to_csv(os.path.join('results', model_name + sname + '_mrr.tsv'), index=False, header=None)
