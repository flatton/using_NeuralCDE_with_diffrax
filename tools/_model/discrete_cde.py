from typing import Tuple, Callable, Optional

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .vector_field import Func

class DiscreteCDECell(eqx.Module, strict=True):
    mlp: Func

    def __init__(self, input_size: int, hidden_size: int, width_size: int, depth: int, use_bias: bool = True, dtype=None, *, key: PRNGKeyArray):
        self.mlp = Func(input_size, hidden_size, width_size, depth, key=key)

    def __call__(self, xi: Array, yi0: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        xi0, xi1 = xi
        h_f = self.mlp(None, yi0)
        yi1 = yi0 + h_f @ (xi1 - xi0)
        return yi1

class DiscreteCDELayer(eqx.Module):
    cell: DiscreteCDECell

    def __init__(self, input_size: int, hidden_size: int, width_size: int, depth: int, *, key: PRNGKeyArray):
        self.cell = DiscreteCDECell(input_size, hidden_size, width_size, depth, key=key)

    def __call__(self, y0: Array, xs1: Array, key: Optional[PRNGKeyArray] = None) -> Tuple[Array, Array]:
        def _f(carry, xs):
            carry = self.cell(xs, carry)
            return carry, carry
        xs1 = jnp.expand_dims(xs1, axis=1)
        xs0 = jnp.zeros_like(xs1)
        xs0 = xs0.at[1:, :, :].set(xs1[:-1, :, :])
        xs = jnp.concatenate([xs0, xs1], axis=1)
        yT, ys = lax.scan(_f, y0, xs)
        return yT, ys

class RNN(eqx.Module):
    hidden_size: int
    initial: eqx.nn.MLP
    rnn: DiscreteCDELayer
    linear: eqx.nn.Linear
    bias: jax.Array
    activation_output: Callable

    def __init__(self, in_size: int, out_size: int, hidden_size: int, width_size: int, depth: int, *, key: PRNGKeyArray):
        ikey, gkey, lkey = jr.split(key, 3)
        self.hidden_size = hidden_size
        self.initial = eqx.nn.MLP(in_size, hidden_size, width_size, depth, key=ikey)
        
        self.rnn = DiscreteCDELayer(in_size, hidden_size, width_size, depth, key=gkey)
        
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)
        self.activation_output = jnn.sigmoid if out_size == 1 else jnn.log_softmax

    def __call__(self, xs: Array, key: Optional[PRNGKeyArray] = None) -> Array:
        x0 = xs[0,:]
        y0 = self.initial(x0) # y0
        yT, ys = self.rnn(y0, xs)
        logits = self.linear(yT)
        # sigmoid because we're performing binary classification
        probs = self.activation_output(logits + self.bias)
        return probs