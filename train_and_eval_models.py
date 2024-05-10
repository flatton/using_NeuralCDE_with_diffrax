import json
import time
import datetime
import gc
from typing import Sequence, Tuple, Union, Callable, Optional

import jax
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray
import optax  # https://github.com/deepmind/optax
import equinox as eqx
from memray import Tracker

from .tools._dataset.datasets import MNISTStrokeDataset
from .tools._dataset.dataloader import dataloader_ununiformed_sequence
from .tools._model.neural_cde import NeuralCDE
from .tools._model.discrete_cde import RNN
from .tools._loss.cross_entropy import bce_loss, nll_loss
from .tools.config import ExperimentConfig
from .tools._train.update import Trainer, run_train
from .tools._eval.evaluation import run_eval

def experiment(
    *,
    dataset_size_train: int,
    dataset_size_test: int,
    noise_ratio: float,
    input_format: str,
    interpolation: str,
    neural_model_name: str,
    out_size: int,
    hidden_size: int,
    width_size: int,
    depth: int,
    batch_size: int,
    lr: float,
    steps: int,
    seed: int,
    output_model_path: str,
) -> Tuple[float, float, float]:
    key = jr.PRNGKey(seed)
    train_data_key, test_data_key, train_model_key, test_model_key, train_run_key, test_run_key = jr.split(key, 6)
    experiment_result = {}

    # Load the train dataset
    dataset = MNISTStrokeDataset(dataset_size=dataset_size_train, mode_train=True, input_format=input_format, noise_ratio=noise_ratio, interpolation=interpolation, key=train_data_key)
    ts, _, coeffs, labels, in_size = dataset.make_dataset()

    # Initialize the model
    if neural_model_name == 'NeuralCDE':
        Model = NeuralCDE
    elif neural_model_name == 'RNN':
        Model = RNN
    model = Model(in_size, out_size, hidden_size, width_size, depth, interpolation=interpolation, key=train_model_key)

    # Choice loss function
    if out_size == 2:
        loss_func = bce_loss
    elif out_size > 2:
        loss_func = nll_loss
    else:
        raise ValueError(f'The `out_size` must be greater than or equal to 2. But now {out_size}')

    # Training
    model, train_time_avg = run_train(steps, batch_size, (ts, *coeffs, labels), model, loss_func, optax.adam(lr), out_size, key=train_run_key)
    experiment_result['train_time_avg'] = float(train_time_avg)

    # Save the model
    eqx.tree_serialise_leaves(output_model_path, model)

    # Clear caches
    del model, dataset, ts, coeffs, labels
    eqx.clear_caches()
    jax.clear_caches()
    gc.collect()
    
    # Load the test dataset
    dataset = MNISTStrokeDataset(dataset_size=dataset_size_test, mode_train=False, input_format=input_format, noise_ratio=noise_ratio, interpolation=interpolation, key=test_data_key)
    ts, _, coeffs, labels, _ = dataset.make_dataset()

    # Load the trained model
    model = eqx.filter_eval_shape(Model, in_size, out_size, hidden_size, width_size, depth, interpolation=interpolation, key=test_model_key)
    model = eqx.tree_deserialise_leaves(output_model_path, model)

    # Evaluation
    test_loss_avg, test_acc_avg, test_time_avg = run_eval((ts, *coeffs, labels), model, loss_func, out_size, key=test_run_key)
    experiment_result['test_loss_avg'] = float(test_loss_avg)
    experiment_result['test_acc_avg'] = float(test_acc_avg)
    experiment_result['test_time_avg'] = float(test_time_avg)
    
    return experiment_result

def main() -> None:
    eqx.clear_caches()
    jax.clear_caches()
    gc.collect()
    
    config = ExperimentConfig()
    
    config.steps = 10000
    config.dataset_size_train = -1
    config.dataset_size_test = -1
    
    config.noise_ratio = 0.
    config.neural_model_name = 'NeuralCDE'
    
    date = datetime.datetime.now().strftime('%Y年%m月%d日-%H:%M:%S')
    config.output_model_name = f'/{config.neural_model_name}-{date}'

    include_fields = [key for key in config.model_dump() if not 'output_' in key] + ['output_model_path',]
    experiment_condition = config.model_dump(include={*include_fields})
    with Tracker(config.output_memray_path):
        experiment_result = experiment(**experiment_condition)

    with open(config.output_config_path, "w") as o:
        print(config.model_dump_json(indent=4), file=o)

    experiment_result['date'] = date
    experiment_result['neural_model_name'] = config.neural_model_name
    experiment_result['interpolation'] = config.interpolation
    experiment_result['noise_ratio'] = config.noise_ratio

    with open(config.output_result_path, mode="w", encoding="utf-8") as o:
        json.dump(experiment_result, o, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()