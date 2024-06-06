from typing import Union, Tuple, Callable, Optional

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import diffrax
import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Array, PRNGKeyArray

from .vector_field import Func

class NeuralCDE(eqx.Module):
    """
        Neural CDE モデルを定義.
    """
    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear
    activation_output: Callable
    interpolation: str

    def __init__(self, in_size: int, out_size: int, hidden_size: int, width_size: int, depth: int, *, interpolation: str = 'cubic', key: PRNGKeyArray, **kwargs):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3) # 乱数生成キーを分割し, 独立した新たなキーを作成
        self.interpolation = interpolation

        self.initial = eqx.nn.MLP(in_size, hidden_size, width_size, depth, key=ikey) # 初期条件のモデルを初期化

        self.func = Func(in_size, hidden_size, width_size, depth, key=fkey) # ベクトル場のモデルを初期化

        self.linear = eqx.nn.Linear(hidden_size, out_size, key=lkey) # 出力層のモデルを初期化
        self.activation_output = jnn.sigmoid if out_size == 1 else jnn.log_softmax

    def __call__(self, ts: Float[Array, "*sequence_length"], coeffs: Tuple[Float[Array, "*sequence_length channels+1"], ...], *, evolving_out: bool = False, key: Optional[PRNGKeyArray] = None) -> Union[Float[Array, "out_size"], Float[Array, "*sequence_length out_size"]]:
        '''Solve contious controlled differential equations.
        連続な制御微分方程式を解く.

        **Arguments:**
        - ts: タイムスタンプの系列
        - coeffs: 補間で使用する係数の系列

        **Returns:**
        - `evolving_out=False` の場合は最後のタイムステップにおける予測値. `evolving_out=True` の場合は各タイムステップにおける予測値の系列.

        **NOTE:**
        Each sample of data consists of some timestamps `ts`, and some `coeffs`
        parameterising a control path. These are used to produce a continuous-time
        input path `control`.
        '''

        # 制御信号（パス）の生成
        if self.interpolation == 'linear':
            control = diffrax.LinearInterpolation(ts, coeffs, jump_ts=ts)
        elif self.interpolation == 'cubic':
            control = diffrax.CubicInterpolation(ts, coeffs)

        # Term の生成
        term = diffrax.ControlTerm(self.func, control).to_ode()

        # ソルバー
        solver = diffrax.Tsit5()

        # 最初のステップに使用するステップサイズ
        dt0 = None

        # 積分区間の始点における状態
        y0 = self.initial(control.evaluate(ts[0]))

        # 微分方程式の解を計算
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
            #adjoint=BacksolveAdjoint(), # If you want to reduce memory usage, turn on this.
            saveat=saveat,
        )

        # 出力層の適用
        if evolving_out:
            probs = jax.vmap(lambda y: self.activation_output(self.linear(y))[0])(solution.ys)
        else:
            probs = self.activation_output(self.linear(solution.ys[-1]))

        return probs
