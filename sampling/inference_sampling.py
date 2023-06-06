import sys
sys.path.append("./GRU4Rec")
import numpy as np, pandas as pd
import gru4rec
import evaluation_utils, test_item_file_builder
import os
import joblib
import argparse

def recall_mrr_from_ranks(ranks, nsample):
	ranks_df = pd.DataFrame({"ranks": ranks.flatten()}).groupby("ranks", as_index=False).agg(sizes=('ranks', 'count'))
	ranks_df.sort_values(by=["ranks"],inplace=True)
	n = nsample + 1
	measures = ranks_df.sizes.sum()
	recall = np.zeros(n)
	recall[ranks_df.ranks.values-1] = ranks_df.sizes.values
	recall = np.cumsum(recall) / measures
	mrr = np.zeros(n)
	mrr[ranks_df.ranks.values-1] = ranks_df.sizes.values / ranks_df.ranks.values
	mrr = np.cumsum(mrr) / measures
	for i in [0,4,9,19]:
		if len(recall)-1 < i:
			break
		print(f"\tRecall@{i+1}\t", f"{recall[i]:.8f}")
		print(f"\tMRR@{i+1}\t\t", f"{mrr[i]:.8f}")
	return recall, mrr

def inference_with_sampling(test_file_path, train_path, model_path, methods):
	model_name = os.path.split(model_path)[-1][:-7]

	test = pd.read_csv(test_file_path, sep='\t', dtype={'ItemId':'str'})
	test_path, test_file = os.path.split(test_file_path)
	gru = gru4rec.GRU4Rec.loadmodel(model_path)
	s_random_names = ['full', '0.1', '0.01', '0.001', '100']
	s_sim_names = ["closest", "farthest", "similar", "dissimilar", "uniform", "popular", "invpopular", "popstatic"]
	for sname in  methods: #s_random_names + s_sim_names:
		recall_result_file = os.path.join('data', 'results', 'sampling_results', model_name + '_' + sname + '_recall.tsv')
		mrr_result_file = os.path.join('data', 'results', 'sampling_results', model_name + '_' + sname + '_mrr.tsv')
		if os.path.isfile(recall_result_file) and os.path.isfile(mrr_result_file):
			print(f"Files already exist, SKIPPING:\n{recall_result_file}\n{mrr_result_file}")
			continue
		if not os.path.isfile(os.path.join('data', 'results', 'sampling_results', model_name + '_' + sname + '_rank.pickle')):
			print('Building rank file for {} for {} samples'.format(model_name, sname))
			if sname == 'full':
				ranks = evaluation_utils.get_rank(gru, test, mode='conservative')
			elif sname in s_random_names:
				if sname == '100':
					nsample = 100
				else:
					nsample = int(float(sname) * len(gru.itemidmap))
				np.random.seed(42)
				ranks = evaluation_utils.get_rank_uniform(gru, test, nsample, mode='conservative')
			elif sname in s_sim_names:
				nsample = 100
				negative_items_file = model_name + '_' + sname
				if not os.path.isfile(os.path.join(test_path, negative_items_file + ".tsv")):
					print(f"Building negative item files for: {s_sim_names}")
					test_item_file_builder.create_test_items(gru, n=nsample, train_path=train_path, test_path=test_file_path, out_path_prefix=os.path.join(test_path, model_name))
				ranks = evaluation_utils.get_rank_sampling(gru, test, mode='conservative', negative_items_file=os.path.join(test_path, negative_items_file + ".tsv"))
			else:
				raise ValueError(f"Invalid method name must be chosen from: {s_random_names+s_sim_names} but got {sname}")
			joblib.dump(ranks, os.path.join('data', 'results', 'sampling_results', model_name + '_' + sname + '_rank.pickle'))
		else:
			ranks = joblib.load(os.path.join('data', 'results', 'sampling_results', model_name + '_' + sname + '_rank.pickle'))
		print('Computing recall/mrr for {} for {} samples'.format(model_name, sname))
		if sname == 'full':
			nsample = len(gru.itemidmap)
		elif (sname == '100') or (sname in s_sim_names):
			nsample = 100
		else:
			nsample = int(float(sname) * len(gru.itemidmap))
		recall, mrr = recall_mrr_from_ranks(ranks=ranks, nsample=nsample)
		pd.DataFrame(data=recall).to_csv(recall_result_file, index=False, header=None)
		pd.DataFrame(data=mrr).to_csv(mrr_result_file, index=False, header=None)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_path', type=str)
	parser.add_argument('--train_path', type=str)
	parser.add_argument('--model_path', type=str)
	parser.add_argument('--methods', type=str, nargs='+', default=['full', '0.1', '0.01', '0.001', '100', "closest", "farthest", "similar", "dissimilar", "uniform", "popular", "invpopular", "popstatic"])
	args = parser.parse_args()

	inference_with_sampling(test_file_path=args.test_path, train_path=args.train_path, model_path=args.model_path, methods=args.methods)