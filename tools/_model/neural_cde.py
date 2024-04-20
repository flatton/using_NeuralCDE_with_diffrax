from typing import Optional

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import diffrax
import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .vector_field import Func

class NeuralCDE(eqx.Module):
    """
        Neural CDE モデルを定義.
    """
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear

    def __init__(self, in_size, out_size, hidden_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3) # 乱数生成キーを分割し, 独立した新たなキーを作成
        self.initial = eqx.nn.MLP(in_size, hidden_size, width_size, depth, key=ikey) # 初期条件のモデルを初期化
        self.func = Func(in_size, hidden_size, width_size, depth, key=fkey) # ベクトル場のモデルを初期化
        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey) # 出力層のモデルを初期化

    def __call__(self, ts, coeffs, evolving_out=False) -> Array:
        # Each sample of data consists of some timestamps `ts`, and some `coeffs`
        # parameterising a control path. These are used to produce a continuous-time
        # input path `control`.
        control = diffrax.CubicInterpolation(ts, coeffs) # 制御信号（パス）の生成
        term = diffrax.ControlTerm(self.func, control).to_ode() # Term の生成
        solver = diffrax.Tsit5() # ソルバー
        dt0 = None # 最初のステップに使用するステップサイズ
        y0 = self.initial(control.evaluate(ts[0])) # 積分区間の始点
        if evolving_out:
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)
        solution = diffrax.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=saveat,
        ) # 微分方程式の解を計算
        if evolving_out:
            probs = jax.vmap(lambda y: jnn.sigmoid(self.linear(y))[0])(solution.ys)
        else:
            (probs,) = jnn.sigmoid(self.linear(solution.ys[-1]))
        return probs