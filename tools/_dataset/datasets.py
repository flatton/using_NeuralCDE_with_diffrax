import math
from typing import Tuple

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, PRNGKeyArray

def make_2dspiral_dataset(dataset_size: int, length: int = 100, add_noise: bool = False, *, key: PRNGKeyArray) -> Tuple[Array, Array, Array, Array, int]:
    """
        二次元の時計回り/反時計回り渦のデータを作成. ラベルは時計回りか反時計回りかの二値ラベル.
    """
    theta_key, noise_key, drop_key = jr.split(key, 3) # 乱数生成キーを分割し, 独立した新たなキーを作成

    theta = jr.uniform(theta_key, (dataset_size,), minval=0, maxval=2 * math.pi) # 一様分布の乱数を作成. 0~2πの間で dataset_size 次元(入力チャネル数)の乱数を作成している.
    # print(f'theta: {theta.shape}') --> theta: (256,)

    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1) # ランダムな系列 theta を cos, sin に入力して三角関数をランダムサンプリングし,
    # print(f'y0: {y0.shape}') --> y0: (256, 2)

    ts = jnp.broadcast_to(jnp.linspace(0, 4 * math.pi, length), (dataset_size, length)) # (入力チャネル数, 系列長) という形式のタイムスタンプ系列を作成. `broadcast_to` で入力チャネル数の次元を追加(`reshape`)して, `dataset_size` に合うように `tile` している.
    # print(f'ts: {ts.shape}') --> ts: (256, 100)

    matrix = jnp.array([[-0.3, 2], [-2, -0.3]]) # 渦を作るための回転行列を定義. 原点からの距離を約2倍しつつ, 反時計回りに約 64 度回転させる.
    ys = jax.vmap(
        lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
    )(y0, ts) # まず, `matrix` を `tij` 倍して、 `y0i` と行列-ベクトル積をとっている. 次に, その行列指数関数を計算している. `vmap` はこれの並列化. (これが渦になってるのがよく分かんない!!! けどモデル開発に全然関係ないので別にいい気がする. `jax.vmap` は並列処理用の関数らしい. デフォルトでは, 入力データの次元 0 について並列化するらしい.
    # print(f'ys: {ys.shape}') --> ys: (256, 100, 2)
    ys = jnp.concatenate([ts[:, :, None], ys], axis=-1) # 隠れ状態のチャネルにタイムスタンプのチャネルを追加
    # print(f'ys: {ys.shape}') --> ys: (256, 100, 3)
    ys = ys.at[: dataset_size // 2, :, 1].multiply(-1) # 入力チャネルの半分までについて, 渦の y 軸の値を正負反転.
    # print(f'ys: {ys.shape}') --> ys: (256, 100, 3)
    if add_noise:
        ys = ys + jr.normal(noise_key, ys.shape) * 0.01 # 全部のチャネルに白色ノイズを付与

    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)
    #print(f'coeffs: {[len(x) for x in coeffs]}') --> coeffs: [256, 256*99*3, 256*99*3, 256*99*3]
    labels = jnp.zeros((dataset_size,))
    # print(f'labels: {labels.shape}') --> labels: (256,)
    labels = labels.at[: dataset_size // 2].set(1.0) # JAX の書き方: In JAX, not use ``x[idx] = y``, but use ``x = x.at[idx].set(y)``
    # print(f'labels: {labels.shape}') --> labels: (256,)
    _, _, in_size = ys.shape
    return ts, ys, coeffs, labels, in_size
