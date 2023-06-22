# recsys_eval_flaws
Experiments of the "Widespread flaws in offline evaluation of recommender systems" paper

## Setup: Re​pository
1. Clone the repository.
​
2. In the root folder of the repository, run the following command to initialize the submodules:
    ```shell
    git submodule update --init --recursive
    ```
    - 📝 NOTE: If the submodules would get some major update and pulling the latest versions would be necessary, run the following command:
        ```shell
        git submodule update --remote --recursive
        ```
## Setup: Environments
1. Install [NVIDIA Driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#)
    - make sure the driver [supports](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) CUDA Toolkit version 11.3.1
    - [*cuda-drivers*](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation) is enough, but installing both [*cuda-drivers* and *cuda-toolkit*](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) is correct as well (the environemnt described in the following steps, will install cudatoolkit, and makes sure it uses the environemnt's cudatoolkit instead of a global version).

2. Install Anaconda by following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation). You can choose either Anaconda or Miniconda.

3. Conda Environments 
    - For any jupyter-notebook (experiments, vsknn train / test, GRU4Rec train / test, etc.) you can use the environment described in `conda_eval_flaws_env.yml`. To create the **eval_flaws** environment, run the following command:
        ```shell
        conda env create -f eval_flaws_env.yml
        ```
    - GRU4Rec experiments require the **gru4rec_theano_gpu** environment.  Training and Testing will be ran automatically when necessary at the beginning of the experiment notebooks. (Training and testing can be executed manually as well, from the *gru4rec_theano_gpu* environment.) To create the environment, please run:
        ```shell
        bash conda_gru4rec_theano_gpu_install.sh
        ```
        - Installation using `conda_gru4rec_theano_gpu_install.sh` is strongly advised, but the environment can also be directly created with conda from the `conda_gru4rec_theano_gpu.yml`. However the installation script makes some extra steps to ensure the environment uses the correct cudatoolkit (which is installed by the environemnt, avoiding collusion with the system cudatoolkit), in this case, these steps must be perfomed manually (refer to `conda_gru4rec_theano_gpu_install.sh` for the exact steps).
    - 📝 NOTE: The installation process might take a few minutes, it is normal. 
    - 📝 NOTE: For the notebooks the **eval_flaws** environment should be used. When gru4rec training or testing scripts are executed automatically from the notebooks, it is ensured that they will be executed from the **gru4rec_theano_gpu** environment.

## Usage
Run the notebooks

## Download link to the datasets

- Rees46:
    - [Official Website](https://rees46.com/en/open-cdp)
    - [Kaggle Dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- Coveo:
    - [Official Download Link](https://www.coveo.com/en/ailabs/shopper-intent-prediction-from-clickstream-e-commerce-data-with-minimal-browsing-information)
- Retailrocket:
    - [Kaggle Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- Amazon Beauty:
    - [Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
    - [Download Link](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv)
- MovieLens 10M:
    - [Official Website](https://grouplens.org/datasets/movielens/)
    - [Official Download Link](https://files.grouplens.org/datasets/movielens/ml-10m.zip)
- Steam Games:
    - [Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data)
    - [Download Link](https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz)
- Yelp:
    - [Official Website](https://www.yelp.com/dataset)
    - [Official Download Link](https://www.yelp.com/dataset/download)