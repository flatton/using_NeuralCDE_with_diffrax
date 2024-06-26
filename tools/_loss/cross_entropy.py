from typing import Optional, Sequence, Tuple, Union

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax  # https://github.com/deepmind/optax
from jaxtyping import Array, Float, PRNGKeyArray


@eqx.filter_jit
def bce_loss(
    model: eqx.Module,
    inputs: Tuple[Float[Array, "batchsize *"], ...],
    labels: Float[Array, "batchsize"],
    *,
    key: PRNGKeyArray,
) -> Tuple[float, float]:
    """Calcurate Binary Cross-Entropy Loss
    Binary Cross-Entropy (;BCE) Loss を算出する関数.

    **Arguments:**
    - inputs: タイムスタンプ `ts` や係数 `coeffs` のタプル.
    - labels: バッチサイズ.
    - key: a pseudo-random number generator (;PRNG) key. `key = jax.random.PRNGKey(seed)`

    **Returns:
    - bxe: Binary Cross-Entropy
    - acc: Accuracy
    """
    batched_keys = jr.split(key, num=labels.shape[0])
    preds = jax.vmap(model, axis_name="batch")(*inputs, key=batched_keys)
    preds = jnp.squeeze(preds)

    # Binary cross-entropy
    bxe = labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds)
    bxe = -jnp.mean(bxe)
    acc = jnp.mean((preds > 0.5) == (labels == 1))
    return bxe, acc


@eqx.filter_jit
def nll_loss(
    model: eqx.Module,
    inputs: Tuple[Float[Array, "batchsize *"], ...],
    labels: Float[Array, "batchsize"],
    out_size: int,
    *,
    key: PRNGKeyArray,
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """Calcurate Negative Log-Likelyfood Loss
    Negative Log-Likelyfood (;NLL) Loss を算出する関数.

    **Arguments:**
    - inputs: タイムスタンプ `ts` や係数 `coeffs` のタプル.
    - labels: バッチサイズ.
    - key: a pseudo-random number generator (;PRNG) key. `key = jax.random.PRNGKey(seed)`

    **Returns:
    - bxe: Binary Cross-Entropy
    - acc: Accuracy
    """
    batched_keys = jr.split(key, num=labels.shape[0])
    preds = jax.vmap(model, axis_name="batch")(*inputs, key=batched_keys)

    # Negative log likelyfood loss
    labels = jnp.squeeze(labels)
    onehot = jnn.one_hot(labels, out_size)
    plogq = jnp.multiply(preds, onehot)
    xe = jnp.sum(plogq, axis=-1)
    xe = -jnp.mean(xe)
    acc = jnp.mean(labels == jnp.argmax(preds, axis=-1))
    return xe, acc


@eqx.filter_jit
def nll_loss_state(
    model: eqx.Module,
    inputs: Tuple[Float[Array, "batchsize *"], ...],
    labels: Float[Array, "batchsize"],
    out_size: int,
    *,
    key: PRNGKeyArray,
) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """Calcurate Negative Log-Likelyfood Loss
    Negative Log-Likelyfood (;NLL) Loss を算出する関数.
    ただし, `equinox.nn.BatchNorm` などの Normalization を使用するために `inputs` には `equinox.nn.State` object が含まれている必要がある.

    **Arguments:**
    - inputs: タイムスタンプ `ts` や係数 `coeffs`, および `State` のタプル.
    - labels: バッチサイズ.
    - key: a pseudo-random number generator (;PRNG) key. `key = jax.random.PRNGKey(seed)`

    **Returns:
    - bxe: Binary Cross-Entropy
    - acc: Accuracy
    - state: an `equinox.nn.State` object
    """
    _axes = tuple([0 for _ in jnp.arange(len(inputs) - 1)]) + (None,)
    batched_keys = jr.split(key, num=labels.shape[0])
    preds, state = jax.vmap(model, axis_name="batch", in_axes=_axes, out_axes=_axes)(
        *inputs, key=batched_keys
    )

    # Negative log likelyfood loss
    labels = jnp.squeeze(labels)
    onehot = jnn.one_hot(labels, out_size)
    plogq = jnp.multiply(preds, onehot)
    xe = jnp.sum(plogq, axis=-1)
    xe = -jnp.mean(xe)
    acc = jnp.mean(labels == jnp.argmax(preds, axis=-1))
    return xe, acc, state
