import gym
import pickle
import flax
import jax
import jax.numpy as jnp
import numpy as np

import os
import pathlib
import functools

from util.dataset import INFINITE, Dataset, DatasetIterator
from util.models import MLP

class GymGenerator(Dataset):
    def __init__(self, rng_key, traj_length, env, policy, 
                    scale_dynamics, vis):
        self._rng_key = rng_key
        self._traj_length = traj_length
        self._env = env
        self._policy = policy
        self._scale_dynamics = scale_dynamics
        self._vis = vis

    @property
    def length(self):
        return INFINITE

    def with_key(self, rng_key):
        return GymGenerator(rng_key, self._traj_length, 
                self._env, self._policy, self._scale_dynamics, self._vis)

    def with_length(self, length):
        return GymGenerator(self._rng_key, length, self._env, self._policy, 
                            self._scale_dynamics, self._vis)

    def with_policy(self, policy):
        return GymGenerator(self._rng_key, self._traj_length, self._env, policy,
                            self._scale_dynamics, self._vis)

    def with_visualize(self, vis):
        return GymGenerator(self._rng_key, self._traj_length, self._env, self._policy, 
                            self._scale_dynamics, vis)

    def iter(self):
        return GymIterator(self, self._rng_key)

class GymIterator(DatasetIterator):
    def __init__(self, gen, key):
        self.key = key
        self.gen = gen

    @property
    def has_next(self):
        return True

    def next(self):
        env = self.gen._env
        reset_key, traj_key, next_key = jax.random.split(self.key, 3)
        traj_keys = jax.random.split(traj_key, self.gen._traj_length)
        env.seed(jax.random.randint(reset_key, (), 0, 100000000).item())
        x =  self.gen._env.reset()
        xs = []
        us = []
        rs = []
        imgs = []
        low, high = env.action_space.low, env.action_space.high
        for i in range(self.gen._traj_length):
            xs.append(x)
            if self.gen._vis:
                imgs.append(env.render(mode='rgb_array'))
            u = self.gen._policy(x, traj_keys[i])
            if self.gen._scale_dynamics:
                u = jax.nn.tanh(u)
                u = low + (0.5 * (u + 1.0) * (high - low))
            u = jnp.clip(u, low, high)
            x, r, _, _ = env.step(np.array(u))
            rs.append(r)
            us.append(u)
        xs, us, rs = jnp.stack(xs), jnp.stack(us), jnp.stack(rs)
        traj = {"x": xs, "u": us, "r": rs}
        if self.gen._vis:
            traj["img"] = jnp.stack(imgs)
        return GymIterator(self.gen, next_key), traj


def load_gym_expert_from_path(path, env, scale_output):
    expert_data = pickle.load(open(path, "rb"))
    expert_data = flax.core.frozen_dict.freeze(expert_data)

    expert_params = jax.device_put(expert_data['params'])

    model = MLP((expert_params['layer_0']['bias'].shape[0], 
        expert_params['layer_1']['bias'].shape[0],
        expert_params['layer_2']['bias'].shape[0]), expert_data['activation'])
    expert_params = flax.core.frozen_dict.freeze({
        'params': expert_params
    })


    use_norm = expert_data['norm'] is not None
    if expert_data['norm'] is not None:
        norm_mean = expert_data['norm']['mean']
        norm_var = expert_data['norm']['var']
        norm_epsilon = expert_data['norm']['epsilon']
        norm_clip = expert_data['norm']['clip']

    low, high = env.action_space.low, env.action_space.high
    @jax.jit
    def expert(x, rng=None):
        if use_norm:
            normed = (x - norm_mean) / jnp.sqrt(norm_var + norm_epsilon)
            x = jnp.clip(normed, -norm_clip, norm_clip)
        u = model.apply(expert_params, x)
        if scale_output:
            u = jax.nn.tanh(u)
            u = low + (0.5 * (u + 1.0) * (high - low))
        return u
    return expert, expert_params

def make_gym_expert_with_params(expert_name, env, scale_output):
    experts_dir = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "experts")
    path = os.path.join(experts_dir, f"{expert_name}.pk")
    return load_gym_expert_from_path(path, env, scale_output)

def make_gym_expert(expert_name, env, scale_output):
    return make_gym_expert_with_params(expert_name, env, scale_output)[0]

def make_gym_policy(rng_key, env_name, env, scale_output):
    u_dim = env.action_space.shape[0]
    x_dim = env.observation_space.shape[0]
    model = MLP((512, 512, u_dim), 'gelu', batch_norm=False)
    init_params = model.init(rng_key, jnp.zeros(x_dim), use_running_average=False)

    low, high = env.action_space.low, env.action_space.high
    @functools.partial(jax.jit, static_argnums=3)
    def policy(vars, x, rng=None, update_batch_stats=False):
        if update_batch_stats:
            u, new_batch_stats = model.apply(vars, x, mutable=['batch_stats'], use_running_average=False)
        else:
            u = model.apply(vars, x, use_running_average=True)
        if scale_output:
            u = jax.nn.tanh(u)
            u = low + (0.5 * (u + 1.0) * (high - low))
        return (u, new_batch_stats) if update_batch_stats else u
    return policy, init_params