# recsys_eval_flaws
Experiments of the "Widespread flaws in offline evaluation of recommender systems" paper

## Usage
To get started with this repository, follow the instructions below:
### Setup Re​positories
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
## Setup Environments
1. Install [NVIDIA Driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#)
    - make sure the driver [supports](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) CUDA Toolkit version up to 11.3.1
    - [*cuda-drivers*](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation) is enough, but installing both [*cuda-drivers* and *cuda-toolkit*](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) is correct as well (the environemnt described in the following steps, will install cudatoolkit, and makes sure it uses the environemnt's cudatoolkit instead of a global version).

2. Install Anaconda by following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation). You can choose either Anaconda or Miniconda.

3. Conda Environments 
    - For any jupyter-notebook (visualization, vsknn experiments etc., except notebooks running GRU4Rec experiments) you can use the environment described in `conda_eval_flaws_env.yml`. To create the **eval_flavs** environment, run the following command:
        ```shell
        conda env create -f eval_flaws_env.yml
        ```
    - GRU4Rec experimnts (training, testing, notebooks) can be run from the **gru4rec_theano_gpu** environment. To create the environment, please run:
        ```shell
        bash conda_gru4rec_theano_gpu_install.sh
        ```
        - Installing with `conda_gru4rec_theano_gpu_install.sh` is strongly advised, but the environment can also be directly created with conda from the `conda_gru4rec_theano_gpu.yml`. However the installation script makes some extra steps to ensure the environment uses the correct cudatoolkit (which is installed by the environemnt, avoiding collusion with the system cudatoolkit), in this case, these steps must be perfomed manually (refer to `conda_gru4rec_theano_gpu_install.sh` for the exact steps). 
    - 📝 NOTE: The installation process might take a few minutes, it is normal. 

## Download link to the datasets

- Rees46:
    - [Official Website](https://rees46.com/en/open-cdp)
    - [Kaggle Dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
​
- Coveo:
    - [Official Download Link](https://www.coveo.com/en/ailabs/shopper-intent-prediction-from-clickstream-e-commerce-data-with-minimal-browsing-information)
​
- Retailrocket:
    - [Kaggle Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)