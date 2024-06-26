from typing import Callable, Optional, Tuple, Union

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlp = Func(input_size, hidden_size, width_size, depth, key=key)

    def __call__(
        self,
        xi: Float[Array, "* input_size"],
        yi0: Float[Array, "hidden_size"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "hidden_size"]:
        xi0, xi1 = xi
        h_f = self.mlp(None, yi0)
        yi1 = yi0 + h_f @ (xi1 - xi0)
        return yi1


class DiscreteCDELayer(eqx.Module):
    cell: DiscreteCDECell

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cell = DiscreteCDECell(input_size, hidden_size, width_size, depth, key=key)

    def __call__(
        self,
        y0: Float[Array, "*sequence_length hidden_size"],
        xs1: Float[Array, "*sequence_length input_size"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[Float[Array, "hidden_size"], Float[Array, "*sequence_length hidden_size"]]:
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
    activation_output: Callable
    interpolation: str

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        *,
        interpolation: str = "cubic",
        key: PRNGKeyArray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ikey, rkey, lkey = jr.split(key, 3)
        self.interpolation = interpolation
        self.hidden_size = hidden_size

        self.initial = eqx.nn.MLP(in_size, hidden_size, width_size, depth, key=ikey)

        self.rnn = DiscreteCDELayer(in_size, hidden_size, width_size, depth, key=rkey)

        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey)
        self.activation_output = jnn.sigmoid if out_size == 1 else jnn.log_softmax

    def __call__(
        self,
        ts: Float[Array, "*sequence_length"],
        coeffs: Tuple[Float[Array, "*sequence_length channels+1"], ...],
        *,
        evolving_out: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Union[Float[Array, "out_size"], Float[Array, "*sequence_length out_size"]]:
        """RNNs as discretized controlled differential equations.
        離散化された制御微分方程式の形式になっているRNN.

        **Arguments:**
        - ts: タイムスタンプの系列
        - coeffs: 補間で使用する係数の系列

        **Returns:**
        - `evolving_out=False` の場合は最後のタイムステップにおける予測値. `evolving_out=True` の場合は各タイムステップにおける予測値の系列.
        """

        # 制御信号（パス）の生成
        if self.interpolation == "linear":
            control = diffrax.LinearInterpolation(ts, coeffs, jump_ts=ts)
        elif self.interpolation == "cubic":
            control = diffrax.CubicInterpolation(ts, coeffs)

        # 補間された入力信号の生成
        x0 = control.evaluate(ts[0])
        delta = jax.vmap(control.evaluate)(ts[:-1], ts[1:])

        def _f(_xi0, _delta):
            _xi1 = _xi0 + _delta
            return _xi1, _xi1

        _, xs = lax.scan(_f, x0, delta)
        xs = jnp.concatenate((x0[None, :], xs), axis=0)

        # 初期状態
        x0 = xs[0, :]
        y0 = self.initial(x0)  # y0

        # RNNブロック
        yT, ys = self.rnn(y0, xs)

        # 出力層
        if evolving_out:
            probs = jax.vmap(lambda y: self.activation_output(self.linear(y))[0])(ys)
        else:
            probs = self.activation_output(self.linear(yT))
        return probs
