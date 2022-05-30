import jax.random
import jax.tree_util as tree_util
import jax.numpy as jnp
import numpy as np
import flax
import hashlib
from util.models import safe_norm
from util.dataset import Dataset, DatasetIterator, INFINITE
from util.dataset.config import fun as fun_conf

class TrajectoryGenerator(Dataset):
    def __init__(self, rng_key, traj_length, 
                init_gen, dynamics, policy):
        assert traj_length > 0
        self._rng_key = rng_key
        self._traj_length = traj_length
        self._init_gen = init_gen
        self._dynamics = dynamics
        self._policy = policy

    @property
    def length(self):
        return INFINITE

    def __config__(self):
        # TODO: Make init_gen, dynamics, and policy have config
        return {'rng_key': self._rng_key,
                'traj_length': self._traj_length,
                'init_gen': self._init_gen.__config__(),
                'dynamics': self._dynamics.__config__(),
                'policy': self._policy.__config__() }

    def with_length(self, length):
        return TrajectoryGenerator(self._rng_key, length, 
                self._init_gen, self._dynamics, self._policy)

    def with_key(self, rng_key):
        return TrajectoryGenerator(rng_key, self._traj_length, 
                self._init_gen, self._dynamics, self._policy)

    def with_policy(self, policy):
        return TrajectoryGenerator(self._rng_key, self._traj_length, 
                self._init_gen, self._dynamics, policy)
    
    def with_visualize(self, vis):
        return self

    def _generate(self, rng_key):
        rng_keys = jax.random.split(rng_key, self._traj_length)
        def scan_fn(state, rng_key):
            p_rng, d_rng = jax.random.split(rng_key)
            inp = self._policy(state, p_rng)
            next_state = self._dynamics(d_rng, state, inp)
            return next_state, (next_state, inp)

        init_key, final_inp_key = jax.random.split(rng_keys[0])
        x0 = self._init_gen(init_key)
        _, (xs, us) = jax.lax.scan(scan_fn, x0, rng_keys[1:])
        xs = jnp.concatenate((jnp.expand_dims(x0,0), xs))
        uf = self._policy(xs[-1], final_inp_key)
        us = jnp.concatenate((us, jnp.expand_dims(uf,0)))
        return {'x': xs, 'u': us}
    
    def iter(self):
        return TrajectoryIterator(self, self._rng_key)
from util.timer import timed

class TrajectoryIterator(DatasetIterator):
    def __init__(self, generator, key, allocated=None, allocated_idx=0):
        self.generator = generator
        self.key = key
        self.allocated = allocated
        self.allocated_idx = allocated_idx

    @property
    def has_next(self):
        return True
    
    def next(self):
        key, sk = jax.random.split(self.key)
        if self.allocated is None or self.allocated_idx >= 64:
            keys = jax.random.split(sk, 64)
            trajs = jax.vmap(self.generator._generate)(keys)
            allocated = trajs
            allocated_idx = 0
        else:
            allocated = self.allocated
            allocated_idx = self.allocated_idx
        x = jax.tree_util.tree_map(lambda x: x[allocated_idx], allocated)
        return TrajectoryIterator(self.generator, key, allocated, allocated_idx + 1), x
    
    def skip(self, n):
        # TODO: This doesn't actually skip n, just muddles the key a bit
        _, sk = jax.random.split(self.key)
        return TrajectoryIterator(self.generator, sk)

class NormalGenerator:
    def __init__(self, state_dim):
        self._state_dim = state_dim

    def __config__(self):
        return {'state_dim': self._state_dim}

    def __call__(self, rng):
        return jax.random.normal(rng, shape=(self._state_dim,))

class ModelPolicy:
    def __init__(self, model, params):
        self._model = model
        self._params = params

    def __config__(self):
        params_hash = jax.tree_util.tree_map(lambda x: hashlib.sha256(np.array(x)).hexdigest(), self._params)
        return {'model': hashlib.sha256(repr(self._model).encode('utf-8')).hexdigest(), 
                'params': params_hash }
    
    def __call__(self, state, rng=None, update_batch_stats=False):
        if update_batch_stats:
            return self._model.apply(self._params, state, mutable=['batch_stats'], use_running_average=False)
        else:
            return self._model.apply(self._params, state, use_running_average=True)

class ArtificialSystem:
    def __init__(self, p, eta, gamma_power, expert):
        assert eta < 4/(5 + p)  # needed for IGS stability
        self.p = p
        self.eta = eta
        self.gamma_power = gamma_power
        self.expert = expert

    def __config__(self):
        return {'p': self.p, 'eta': self.eta, 'gamma_power': self.gamma_power, 'expert': self.expert.__config__()}
    
    def __call__(self, rng, x, u):
        p = self.p
        eta = self.eta
        diff = u - self.expert(x)
        diff_norm = jnp.linalg.norm(diff)
        scale = (diff_norm + 0.000001)**(self.gamma_power - 1)

        f = x - eta * x * (jnp.abs(x) ** p) / (1 + (jnp.abs(x)**p))
        g = eta / (1 + (jnp.abs(x)**p)) * scale*diff
        return f + g

class ExponentialSystem:
    def __init__(self, decay, c, gamma_power, expert):
        assert decay < 1
        self.decay = decay
        self.c = c
        self.gamma_power = gamma_power
        self.expert = expert

    def __config__(self):
        return {'p': self.p, 'eta': self.eta, 'gamma_power': self.gamma_power, 'expert': self.expert.__config__()}
    
    def __call__(self, rng, x, u):
        diff = (u - self.expert(x))
        diff_norm = safe_norm(diff, 1e-6)
        # Do a minimum to prevent blowup. This will not affect the
        # decay behavior and so is safe to do
        scale = diff_norm**(self.gamma_power - 1)

        f = self.decay * x
        g = self.c * (1 - self.decay) * scale * diff
        return f + g

class PendulumSystem:
    def __init__(self, m, g, l):
        self.m = m
        self.g = g
        self.l = l
    
    def __config__(self):
        return {'m': self.m, 'g': self.g, 'l': self.l}

    def __call__(self, rng, x, u):
        m = self.m
        g = self.g
        l = self.l

@jax.jit
def flatten_trajectory(traj):
    T = traj['x'].shape[0]
    return [ {k: v[i] for k, v in traj.items()} for i in range(T) ]
