# recsys_eval_flaws
Experiments of the "Widespread flaws in offline evaluation of recommender systems" paper

## Usage
​
To get started with this repository, follow the instructions below:
​
1. Clone the repository.
​
2. In the root folder of the repository, run the following command to initialize the submodules:
    ```shell
    git submodule update --init --recursive
    ```

## Environments
1. Install Anaconda by following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation). You can choose either Anaconda or Miniconda.

2. For any other experiment and for the jupyter-notebooks you can use the environment described in `eval_flaws_env.yaml`. To create the environment, run the following command:
    ```shell
    conda env create -f eval_flaws_env.yaml
    ```

3. To run the experiments with GRU4Rec insall the *theano environment* by following the instructions described in the [Official GRU4Rec implementation](https://github.com/hidasib/GRU4Rec/tree/master)

## Experiments
### Ranks
1. Run the following command, replace the parameters to match your files
    ```shell
    THEANO_FLAGS=device=cuda0 python rank_to_metric.py --test_path data/data_sources/coveo_ecommerce/coveo_processed_view_test.tsv --train_path data/data_sources/coveo_ecommerce/coveo_processed_view_train_full.tsv --model_path data/models/coveo_optuna_mrr_bprmax_constrained_fulltrain.pickle 
    ```
2.
    - Use *rank_visualize.ipynb* to create plots for model performance with different sampling techniques measured at different ranks
    - Use *rank_visualize_bars.ipynb* to create bar plots

```shell
THEANO_FLAGS=device=cuda0 python rank_to_metric.py --test_path data/data_sources/diginetica_ecommerce/diginetica_processed_view_7_test.tsv --train_path data/data_sources/diginetica_ecommerce/diginetica_processed_view_7_train_full.tsv --model_path data/models/diginetica_optuna_mrr_bprmax_constrained_fulltrain.pickle
```

### Rules
#### Rule overlap
1. run the following commands
    ```shell
    python rule_overlap.py --full_path data/data_sources/coveo_ecommerce/coveo_processed_view_full.tsv --train_path data/data_sources/coveo_ecommerce/coveo_processed_view_train_full.tsv --test_path data/data_sources/coveo_ecommerce/coveo_processed_view_test.tsv
    ```

    ```shell
    python rule_overlap.py --full_path data/data_sources/retailrocket_ecommerce/retailrocket_processed_view_full.tsv --train_path data/data_sources/retailrocket_ecommerce/retailrocket_processed_view_7a_train_full.tsv --test_path data/data_sources/retailrocket_ecommerce/retailrocket_processed_view_7a_test.tsv
    ```

    ```shell
    python rule_overlap.py --full_path data/data_sources/rees46_ecommerce/rees46_processed_view_userbased_train_full.tsv --train_path data/data_sources/rees46_ecommerce/rees46_processed_view_userbased_train_tr.tsv --test_path data/data_sources/rees46_ecommerce/rees46_processed_view_userbased_test.tsv
    ```

#### New rules
1. asd
    ```shell
    python rule_new_rules.py --full_path data/data_sources/coveo_ecommerce/coveo_processed_view_full.tsv
    ```

    ``` shell
    python rule_new_rules.py --full_path data/data_sources/rees46_ecommerce/rees46_processed_view_userbased_full.tsv --time_units s
    ```
2. bsd

### VSKNN
    ```shell
    python vsknn_optimize.py --train_path data/data_sources/coveo_ecommerce/coveo_processed_view_train_tr.tsv --test_path data/data_sources/coveo_ecommerce/coveo_processed_view_train_valid.tsv --model vsknn --eval_method all --max_iter 100
    ```