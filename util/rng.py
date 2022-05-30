import jax.random
import jax.numpy as jnp

class PRNGSequence:
    def __init__(self, key_or_seed):
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        elif (hasattr(key_or_seed, "shape") and (not key_or_seed.shape) and
                hasattr(key_or_seed, "dtype") and key_or_seed.dtype == jnp.int32):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        self._key = key_or_seed
    
    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n