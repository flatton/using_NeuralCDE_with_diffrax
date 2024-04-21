from typing import Tuple, Sequence, Union, Optional

import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray
import optax  # https://github.com/deepmind/optax


@eqx.filter_jit
def bce_loss(model: eqx.Module, inputs: Tuple[Sequence[Union[Float, Array]], ...], labels: Sequence[Float], *, key: Optional[PRNGKeyArray] = None) -> Tuple[Float, Float]:
    #batched_keys = jr.split(key, num=x.shape[0])
    preds = jax.vmap(model)(*inputs)
    preds = jnp.squeeze(preds)
    
    # Binary cross-entropy
    bxe = labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds)
    bxe = -jnp.mean(bxe)
    acc = jnp.mean((preds > 0.5) == (labels == 1))
    return bxe, acc

@eqx.filter_jit
def nll_loss(model: eqx.Module, inputs: Tuple[Sequence[Union[Float, Array]], ...], labels: Sequence[Float], out_size: int, *, uniform_batch_len: bool = True, key: Optional[PRNGKeyArray] = None) -> Tuple[Float, Float]:
    #batched_keys = jr.split(key, num=x.shape[0])
    preds = jax.vmap(model)(*inputs)
    preds = jnp.squeeze(preds)
    
    # Negative log likelyfood loss
    onehot = jnn.one_hot(labels, out_size)
    xe = jnp.sum(jnp.multiply(preds, onehot), axis=-1)
    xe = -jnp.mean(xe)
    acc = jnp.mean(labels == jnp.argmax(preds, -1))
    return xe, acc