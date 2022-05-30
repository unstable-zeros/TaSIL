from loguru import logger
import os
import tqdm
from util.rng import PRNGSequence
from util.timer import timed
from util.logging import logging_redirect_tqdm

import optax
import jax
import jax.numpy as jnp
import jax.experimental.host_callback

import flax.core.frozen_dict as frozen_dict

class Trainer:
    def __init__(self, opt_init, opt_update, epochs, batch_size, shuffle_buf_size=None):
        self.opt_init = opt_init
        self.opt_update = opt_update
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_buf_size = shuffle_buf_size

    def train(self, rng, dataset, loss_fn, initial_params):
        stats = None
        ibar = None
        def host_callback(s, transforms):
            nonlocal stats
            nonlocal ibar
            stats = s
            ibar.update(1)
        @jax.jit
        def iter_fn(key, opt_state, vars, batch, iteration):
            # Extract the params, batch stats
            params = vars['params']
            batch_stats = vars['batch_stats'] if 'batch_stats' in vars else None
            def avg_loss(params, key):
                arg_vars = frozen_dict.freeze({'params': params, 'batch_stats': batch_stats}) if 'batch_stats' in vars else \
                           frozen_dict.freeze({'params': params})
                keys = jax.random.split(key, self.batch_size)
                batch_loss, stats, new_batch_stats = jax.vmap(loss_fn, in_axes=(0, None, 0, None), out_axes=(0, 0, None), axis_name='batch')(keys, arg_vars, batch, iteration)
                output = jnp.mean(batch_loss), (jax.tree_util.tree_map(jnp.mean, stats), new_batch_stats)
                return output
            key, sk = jax.random.split(key)
            grads, (stats, new_batch_stats) = jax.grad(avg_loss, has_aux=True)(params, sk)

            updates, new_opt_state = self.opt_update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            new_vars = frozen_dict.freeze({'params': new_params, 'batch_stats': new_batch_stats['batch_stats']}) if 'batch_stats' in vars else \
                       frozen_dict.freeze({'params': new_params })

            return key, new_opt_state, new_vars, stats
        @jax.jit
        def scan_fn(state):
            key, iterator, opt_state, params, iteration, stats = state
            new_iterator, batch = iterator.next()
            new_key, new_opt_state, new_params, new_stats = iter_fn(key, opt_state, params, batch, iteration)
            # Load the stats to the host
            if stats is not None:
                new_stats = jax.tree_util.tree_map(lambda x, y: 0.2*x + 0.8*y, new_stats, stats)
            new_iterator = jax.experimental.host_callback.id_tap(host_callback, new_stats, result=new_iterator)
            return new_key, new_iterator, new_opt_state, new_params, iteration + 1, new_stats
        @jax.jit
        def train_epoch(key, iterator, opt_state, params, iteration):
            logger.debug("Tracing train_epoch code")
            state = key, iterator, opt_state, params, iteration, None
            # Do the first iteration manually so we get the right stats dictionary for the while loop
            state = scan_fn(state)
            key, _, opt_state, params, iteration, _ = jax.lax.while_loop(lambda s: s[1].has_next, scan_fn, state)
            return key, opt_state, params, iteration

        dataset = dataset.preload()
        opt_state = self.opt_init(initial_params['params'])
        params = initial_params
        rng_seq = PRNGSequence(rng)
        iteration = 0
        with timed("train"):
            with logging_redirect_tqdm():
                with tqdm.tqdm(total=self.epochs) as pbar:
                    for epoch in range(self.epochs):
                        data = dataset.shuffle(next(rng_seq), self.shuffle_buf_size) \
                                      .batch(self.batch_size)
                        with tqdm.tqdm(total=len(data), leave=False) as ibar:
                            iterator = data.iter()
                            _, data = iterator.next()
                            _, opt_state, params, iteration = train_epoch(next(rng_seq), iterator, opt_state, params, iteration)
                        stats_str = ' '.join([ '{}: {: <12g}'.format(k, v) for (k, v) in stats.items() ]) if stats is not None else ''
                        pbar.set_postfix_str(stats_str)
                        pbar.update(1)


        return opt_state, params
    
    @staticmethod
    def eval(key, dataset, eval_fn, batch_size):
        # Turn the entire dataset into one batch
        rng_seq = PRNGSequence(key)
        samples = dataset.length
        dataset = dataset.batch(batch_size)
        it = dataset.iter()
        batch_eval = jax.jit(jax.vmap(eval_fn, in_axes=(0, 0)))
        data_stats = None
        while it.has_next:
            it, data = it.next()
            keys = jax.random.split(next(rng_seq), batch_size)
            batch_stats = batch_eval(keys, data)
            if data_stats is None:
                data_stats = jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), batch_stats)
            else:
                data_stats = jax.tree_util.tree_map(lambda d, b: d + jnp.sum(b, axis=0), data_stats, batch_stats)
        data_stats = jax.tree_util.tree_map(lambda x: (x/samples).item(), data_stats)
        return data_stats