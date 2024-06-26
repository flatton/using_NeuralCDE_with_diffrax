import time
from typing import Callable, Sequence, Tuple

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm.contrib import tzip


def run_eval(
    data: Tuple[Sequence[Float[Array, "sequence *"]], ...],
    model: eqx.Module,
    loss_func: Callable,
    *args,
    key: PRNGKeyArray,
) -> Tuple[float, float, float]:
    # Change the mode of the model into Inference
    model = eqx.nn.inference_mode(model)

    list_xe = []
    list_acc = []
    list_time = []

    ts, *coeffs, labels = data
    (test_step_key,) = jr.split(key, 1)
    for data in tzip(ts, *coeffs, labels):
        test_step_key, _test_step_key = jr.split(test_step_key, 2)
        data = [jnp.expand_dims(x, axis=0) for x in data]
        _ts, *_coeffs, _label = data

        start = time.time()
        xe, acc = loss_func(model, (_ts, _coeffs), _label, *args, key=_test_step_key)
        end = time.time()

        list_xe.append(lax.stop_gradient(xe))
        list_acc.append(lax.stop_gradient(acc))
        list_time.append(end - start)

    loss_avg = jnp.mean(jnp.array(list_xe))
    acc_avg = jnp.mean(jnp.array(list_acc))
    time_avg = jnp.mean(jnp.array(list_time))
    print(f"Test loss: {loss_avg}, Test Accuracy: {acc_avg}, Computation time: {time_avg}")

    return loss_avg, acc_avg, time_avg
