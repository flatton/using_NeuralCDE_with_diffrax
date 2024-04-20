from typing import Optional

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
from jaxtyping import Array, PRNGKeyArray

class Func(eqx.Module):
    """
        ベクトル場を定義.
    """
    mlp: eqx.nn.MLP
    in_size: int
    hidden_size: int

    def __init__(self, in_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * in_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            # Note the use of a tanh final activation function. This is important to
            # stop the model blowing up. (Just like how GRUs and LSTMs constrain the
            # rate of change of their hidden states.)
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t: Optional[Array], y: Array, args: Optional[Array] = None):
        return self.mlp(y).reshape(self.hidden_size, self.in_size)