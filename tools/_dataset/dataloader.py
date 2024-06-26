from typing import Generator, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def dataloader(
    arrays: Tuple[Array, ...], batch_size: int, *, key: PRNGKeyArray
) -> Generator[Tuple[Float[Array, "batchsize"], ...], None, None]:
    """Data loader for data whose sequence length is constant.
    系列長が一定のデータに対するデータローダ.

    **Arguments:**
    - arrays: タイムスタンプ `ts` や係数 `coeffs`, 正解ラベル `labels` のタプル.
    - batch_size: バッチサイズ.
    - key: a pseudo-random number generator (;PRNG) key. `key = jax.random.PRNGKey(seed)`

    **Yeilds:
    - `Tuple[Array, ...]`: タイムスタンプ `ts` や係数 `coeffs`, 正解ラベル `labels` を `batch_size` に切り出した Array のタプル.
    """
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        key, _key = jr.split(key, 2)
        perm = jr.permutation(_key, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def dataloader_ununiformed_sequence(
    arrays: Tuple[Sequence[Union[int, float, Array]], ...], batch_size: int, *, key: PRNGKeyArray
) -> Generator[Tuple[Float[Array, "batchsize"], ...], None, None]:
    """Data loader for data whose sequence length is constant.
    系列長が非一定のデータに対するデータローダ.

    **Arguments:**
    - arrays: タイムスタンプ `ts` や係数 `coeffs`, 正解ラベル `labels` のタプル.
    - batch_size: バッチサイズ.
    - key: a pseudo-random number generator (;PRNG) key. `key = jax.random.PRNGKey(seed)`

    **Yeilds:
    - `Tuple[Array, ...]`: タイムスタンプ `ts` や係数 `coeffs`, 正解ラベル `labels` を `batch_size` に切り出した Array のタプル.
    """
    data, labels = arrays
    dataset_size = len(labels)
    assert all(len(x) == dataset_size for x in data)
    indices = jnp.arange(dataset_size)
    while True:
        key, _key = jr.split(key, 2)
        perm = jr.permutation(_key, indices)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            batched_labels = labels[batch_perm]
            batched_data = tuple([x[idx] for idx in batch_perm] for x in data)

            uniformed_data = tuple()
            for _batch in batched_data:
                max_length = jnp.max(jnp.array([len(x) for x in _batch]))
                uniformed_batch = []
                for _data in _batch:
                    _len = jnp.size(_data, axis=0)
                    _dim = jnp.size(_data, axis=-1)
                    if _len == max_length:
                        uniformed_batch.append(_data)
                    else:
                        if _dim == _len:
                            pad = jnp.tile(_data[-1], max_length - _len)
                            uniformed_batch.append(jnp.append(_data, pad, axis=0))
                        else:
                            pad = jnp.tile(_data[-1], (max_length - _len, 1))
                            uniformed_batch.append(jnp.append(_data, pad, axis=0))
                uniformed_data = uniformed_data + (jnp.array(uniformed_batch),)
            yield uniformed_data + (batched_labels,)
            start = end
            end = start + batch_size
