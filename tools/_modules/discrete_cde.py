from typing import Optional

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .vector_field import Func

class DiscreteCDECell(eqx.Module, strict=True):
    mlp: Func
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        self.mlp = Func(input_size, hidden_size, width_size, depth, key=key)

    def __call__(
        self, xi: Array, yi0: Array, *, key: Optional[PRNGKeyArray] = None
    ):
        xi0, xi1 = xi
        h_f = self.mlp(None, yi0)
        yi1 = yi0 - h_f @ (xi1 - xi0)
        return yi1

class DiscreteCDELayer(eqx.Module):
    cell: DiscreteCDECell
    
    def __init__(self, input_size: int, hidden_size: int, width_size: int, depth: int, *, key: PRNGKeyArray):
        self.cell = DiscreteCDECell(input_size, hidden_size, width_size, depth, key=key)
        
    def __call__(self, y0: Array, xs1: Array, key: Optional[jax.random.PRNGKey] = None):
        def _f(carry, xs):
            carry = self.cell(xs, carry)
            return carry, carry
        xs1 = jnp.expand_dims(xs1, axis=1)
        xs0 = jnp.zeros_like(xs1)
        xs0 = xs0.at[1:, :, :].set(xs1[:-1, :, :])
        xs = jnp.concatenate([xs0, xs1], axis=1)
        yT, ys = lax.scan(_f, y0, xs)
        return yT, ys