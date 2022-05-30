import flax.linen as nn
import flax.linen.linear
import flax.linen.initializers
from typing import Sequence, Callable, Optional, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp

PRNGKey = Any
Shape = Tuple[int,...]
Dtype = Any
Array = Any

class MLP(nn.Module):
    features: Sequence[int]
    activation: str
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = flax.linen.initializers.orthogonal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = flax.linen.initializers.zeros
    batch_norm: bool = False
    use_running_average: Optional[bool] = None

    @nn.compact
    def __call__(self, x, use_running_average=None):
        if not self.batch_norm:
            use_running_average=False
        use_runnning_average = nn.merge_param('use_running_average', self.use_running_average, use_running_average)
        if self.activation == 'relu':
            activation_fn = jax.nn.relu
        elif self.activation == 'gelu':
            activation_fn = jax.nn.gelu
        elif self.activation == 'tanh':
            activation_fn = jax.nn.tanh
        else:
            raise ValueError(f"Expected relu, tanh, or gelu, got {self.activation}")
        for (i, feat) in enumerate(self.features):
            x = nn.Dense(feat, name=f"layer_{i}", kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            if i != len(self.features) - 1:
                if self.batch_norm:
                    x = nn.BatchNorm(use_running_average=use_running_average,
                    momentum=0.9,
                    epsilon=1e-5, axis_name='batch')(x)
                x = activation_fn(x)
        return x

def safe_norm(x, min_norm, *args, **kwargs):
  """Returns jnp.maximum(jnp.linalg.norm(x), min_norm) with correct gradients.
    The gradients of jnp.maximum(jnp.linalg.norm(x), min_norm) at 0.0 is NaN,
    because jax will evaluate both branches of the jnp.maximum.
    The version in this function will return the correct gradient of 0.0 in this
    situation.
    Args:
    x: jax array.
    min_norm: lower bound for the returned norm.
  """
  norm = jnp.linalg.norm(x, *args, **kwargs)
  x = jnp.where(norm < min_norm, jnp.ones_like(x), x)
  return jnp.where(norm < min_norm, min_norm,
                   jnp.linalg.norm(x, *args, **kwargs))


from jax import custom_vjp

@custom_vjp
def clip_gradient(x, lo, hi):
    return x  # identity function

def clip_gradient_fwd(x, lo, hi):
    return x, (lo, hi)  # save bounds as residuals

def clip_gradient_bwd(res, g):
  lo, hi = res
  return (jnp.clip(g, lo, hi), None, None)  # use None to indicate zero cotangents for lo and hi

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

from jax.example_libraries.optimizers import l2_norm

def scale_clip_grads(g, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = safe_norm(g, 1e-9)
  return jnp.where(norm < max_norm, g, g * max_norm / (norm))

@custom_vjp
def scale_clip_bp(x, max_norm):
    return x  # identity function

def scale_clip_bp_fwd(x, max_norm):
    return x, max_norm # save bounds as residuals

def scale_clip_bp_bwd(max_norm, g):
  return (scale_clip_grads(g, max_norm), None)  # use None to indicate zero cotangents for lo and hi

scale_clip_bp.defvjp(scale_clip_bp_fwd, scale_clip_bp_bwd)