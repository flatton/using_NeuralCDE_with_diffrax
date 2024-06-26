import math

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib
import matplotlib.pyplot as plt

from .._dataset import dataloader


def test_module(seed: int = 5678, steps: int = 2, batch_size: int = 2) -> None:
    key = jr.PRNGKey(seed)
    theta_key, loader_key = jr.split(key, 2)

    dataset_size = 4
    sequence_length = 6
    timestamps = jnp.broadcast_to(
        jnp.linspace(0, 4 * math.pi, sequence_length), (dataset_size, sequence_length)
    )

    theta = jr.uniform(theta_key, (dataset_size,), minval=0, maxval=2 * math.pi)
    initail_condition = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    matrix = jnp.array(
        [[-0.3, 2], [-2, -0.3]]
    )  # 渦を作るための回転行列を定義. 原点からの距離を約2倍しつつ, 反時計回りに約 64 度回転させる.
    series_data = jax.vmap(
        lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
    )(initail_condition, timestamps)

    series_data = jnp.concatenate([timestamps[:, :, None], series_data], axis=-1)
    series_data = series_data.at[: dataset_size // 2, :, 1].multiply(-1)

    labels = jnp.zeros((dataset_size,))
    labels = labels.at[: dataset_size // 2].set(1.0)

    for step_i, data_i in zip(
        range(steps), dataloader.dataloader((series_data, labels), batch_size, key=loader_key), strict=False
    ):
        jax.debug.print("series_data-{}: {}", step_i, data_i[0])
        jax.debug.print("labels-{}: {}", step_i, data_i[1])


# %% 実行部
if __name__ == "__main__":
    test_module()
