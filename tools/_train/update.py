import time
from typing import Callable, Tuple, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.example_libraries.optimizers import OptimizerState
from jaxtyping import Array, Float, PRNGKeyArray
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import optax  # https://github.com/deepmind/optax

from tools._dataset.dataloader import dataloader_ununiformed_sequence

class Trainer():
    """Update the model
    モデルの学習処理に関するクラス
    """
    grad_loss: Callable
    optim: optax.GradientTransformation
    
    def __init__(self, *, loss_func: Callable, optimizer: optax.GradientTransformation):
        self.grad_loss = eqx.filter_value_and_grad(loss_func, has_aux=True)
        self.optim = optimizer

    def init_opt_state(self, model: eqx.Module) -> OptimizerState:
        opt_state = self.optim.init(eqx.filter(model, eqx.is_inexact_array))
        return opt_state
    
    @eqx.filter_jit
    def make_step(self, model: eqx.Module, data: Tuple[Float[Array, "batchsize"], ...], opt_state: OptimizerState, *args, key:PRNGKeyArray) -> Tuple[Float, Float, eqx.Module, OptimizerState]:
        ts, *coeffs, labels = data
        (xe, acc), grads = self.grad_loss(model, (ts, coeffs), labels, *args, key=key)
        updates, opt_state = self.optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return xe, acc, model, opt_state

def run_train(steps: int, batch_size: int, data: Tuple[Sequence[Float[Array, "sequence *"]], ...], model: eqx.Module, loss_func: Callable, optimizer: optax.GradientTransformation, *args, key: PRNGKeyArray) -> eqx.Module:
    # Change the mode of the model into Train
    model = eqx.nn.inference_mode(model, value=False)

    # Initialize Trainer
    trainer = Trainer(loss_func=loss_func, optimizer=optimizer)
    opt_state = trainer.init_opt_state(model)

    # Train-steps
    ts, *coeffs, labels = data
    loader_key, train_step_key = jr.split(key, 2)

    list_time = []
    for step, data in zip(
        range(steps), dataloader_ununiformed_sequence(((ts, *coeffs), labels), batch_size, key=loader_key)
    ):
        train_step_key, _train_step_key = jr.split(train_step_key, 2)
        start = time.time()
        xe, acc, model, opt_state = trainer.make_step(model, data, opt_state, *args, key=_train_step_key)
        end = time.time()
        print(f"Step: {step}, Loss: {xe}, Accuracy: {acc}, Computation time: {end - start}")
        list_time.append(end - start)
    time_avg = jnp.mean(jnp.array(list_time))
    
    return model, time_avg