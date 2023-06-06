import sys
sys.path.append("../GRU4Rec")
import os
from utils.experiment_setup import gru4rec_sampling_setups as setups

def create_gru4rec_script(train_path, loss, optim, constrained_embedding, embedding, final_act, layers, batch_size, dropout_p_embed, dropout_p_hidden, learning_rate, momentum, sample_alpha, bpreg, logq, hidden_act, n_epochs, n_sample, test_path=None, save_path=None, model_variant="gru4rec", m=None):
    ps = f"layers={layers},loss={loss},bpreg={bpreg},logq={logq},final_act={final_act},hidden_act={hidden_act},n_epochs={n_epochs},batch_size={batch_size},dropout_p_embed={dropout_p_embed},dropout_p_hidden={dropout_p_hidden},learning_rate={learning_rate},momentum={momentum},sample_alpha={sample_alpha},n_sample={n_sample},constrained_embedding={constrained_embedding},embedding={embedding},adapt={optim},item_key=ItemId,session_key=SessionId,time_key=Time"
    save_path_arg = f" -s {save_path}" if save_path is not None else ""
    m_arg = f" -m {' '.join([str(x) for x in m])}" if m is not None else ""
    test_path_arg = f" -t {test_path}" if test_path is not None else ""
    python_path = "PYTHONPATH=ffn4rec " if model_variant =="ffn4rec" else ""
    script = f"THEANO_FLAGS=device=cuda0 {python_path}python ./GRU4Rec/run.py {train_path}{test_path_arg}{save_path_arg} -g {model_variant} -ps {ps}{m_arg}"
    return script

def train_test_gru4rec_models(experiments, setups, save_model=False, n_epochs=10, m=None, model_variant="gru4rec"):
    for dataset_key, dataset_paths in experiments.items():
        for model_key, params in setups[dataset_key].items():    
            train_path = dataset_paths["train_path"]
            test_path = dataset_paths.get("test_path")
            if test_path is None:
                m = None
            if save_model:
                save_path = os.path.join("data", "models", model_key+".pickle")
                if os.path.isfile(save_path):
                    print(f"SKIPPING training, model already exists: {save_path}")
                    continue
            else:
                save_path = None
            script = create_gru4rec_script(train_path=train_path, test_path=test_path, save_path=save_path, n_epochs=n_epochs, m=m, model_variant=model_variant, **params)
            print(f"executing script:\n{script}")
            ret_code = os.system(script)
            if ret_code != 0:
                raise ValueError(f"Process terminated with exit code: {ret_code}")