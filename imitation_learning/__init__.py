from util.rng import PRNGSequence
from util.models import safe_norm, clip_gradient, scale_clip_bp 
from util.dataset import Dataset
from util.trainer import Trainer
from util.sweep import log_info
from imitation_learning.rollout import ModelPolicy
import util.dataset.config as dconfig

from imitation_learning.bc import bc
from imitation_learning.dagger import dagger
from imitation_learning.dart import dart
from imitation_learning.envs import make_system, make_dataset

from jax.random import PRNGKey
import jax.numpy as jnp

import numpy as np
import jax
import flax
import optax
import imageio

from loguru import logger
import functools

def make_algorithm(algorithm):
    if algorithm == 'bc':
        return bc
    elif algorithm == 'dagger':
        return dagger
    elif algorithm == 'dart':
        return dart

def train(config):
    rng = PRNGSequence(PRNGKey(config.seed))
    d_aux_rng = next(rng)

    # A function to modify the data generator
    # with what we expect
    def data_preprocess(generator):
        # Make sure we have annotated the data with the expert
        @jax.jit
        def add_data(traj, rng):
            traj = dict(traj)
            traj['expert_u'] = jax.vmap(expert)(traj['x'])
            if config.jac_lambda or config.report_jac_error:
                traj['expert_jac'] = jax.vmap(jax.jacrev(expert))(traj['x'])
            if config.hess_lambda or config.report_hess_error:
                traj['expert_hess'] = jax.vmap(jax.jacfwd(jax.jacrev(expert)))(traj['x'])
            traj['rng'] = jax.random.split(rng, traj['x'].shape[0])
            return traj
        generator = generator.map(add_data, d_aux_rng)
        return generator

    # The base loss function
    def loss_fn(key, policy, params, sample, iteration, update_batch_stats=True):
        x = sample['x']
        expert_u = sample['expert_u']

        if update_batch_stats:
            train_u, new_batch_stats = policy(x, update_batch_stats=True)
        else:
            train_u = policy(x)
            new_batch_stats = None

        # pi_error = jnp.sum(jnp.square(train_u - expert_u))
        pi_error = safe_norm(train_u - expert_u, 1e-8)

        loss = pi_error
        stats = {'pi_error': pi_error }
        if config.jac_lambda > 0 or config.report_jac_error:
            expert_jac = sample['expert_jac']
            train_jac = jax.jacrev(policy)(x)
            if config.jac_clip > 0:
                train_jac = scale_clip_bp(train_jac, config.jac_clip)
            # jac_error = jnp.sum(jnp.square(expert_jac - train_jac))
            jac_error = safe_norm(expert_jac - train_jac, 1e-8)
            loss = loss + config.jac_lambda * jac_error
            stats['jac_error'] = jac_error
            stats['jac_t_norm'] = safe_norm(train_jac, 1e-8)
            stats['jac_e_norm'] = safe_norm(expert_jac, 1e-8)

        if config.diff_lambda > 0 or config.report_diff_error:
            state_dim = x.shape[-1]
            if config.diff_orthogonal > 0:
                A = jax.random.normal(key if config.diff_resample else sample['rng'],
                                    (min(config.diff_orthogonal, state_dim), state_dim))
                Q, R = jnp.linalg.qr(A.T)
                diag_sign = jax.lax.broadcast_to_rank(jnp.sign(jnp.diag(R)), rank=Q.ndim)
                Q *= diag_sign # needed for +-
                deltas = Q.T
            elif config.diff_normal > 0:
                A = jax.random.normal(key if config.diff_resample else sample['rng'],
                                        (config.diff_normal, state_dim))
                A_norms = jax.vmap(jnp.linalg.norm)(A)
                deltas = A / jnp.expand_dims(A_norms, -1)
            else:
                deltas = jnp.concatenate((jnp.eye(state_dim),
                        -jnp.eye(state_dim)))
            deltas = jnp.expand_dims(x, -2) + config.diff_eps*deltas
            expert_deltas = jax.vmap(expert)(deltas)
            train_deltas = jax.vmap(policy)(deltas)
            if config.diff_augment:
                aug_error = jnp.sum(jnp.square(train_deltas - expert_deltas))
                loss = loss + config.diff_lambda*aug_error
                stats['aug_error'] = aug_error
            else:
                train_diff_jac = (train_deltas - jnp.expand_dims(train_u, -2))/config.diff_eps
                expert_diff_jac = (expert_deltas - jnp.expand_dims(expert_u, -2))/config.diff_eps
                diff_error = safe_norm(train_diff_jac - expert_diff_jac, 1e-8)
                loss = loss + config.diff_lambda*diff_error
                stats['diff_error'] = diff_error
                stats['diff_t_norm'] = safe_norm(train_diff_jac, 1e-8)
                stats['diff_e_norm'] = safe_norm(expert_diff_jac, 1e-8)


        if config.hess_lambda > 0 or config.report_hess_error:
            expert_hess = sample['expert_hess']
            train_hess = jax.jacfwd(jax.jacrev(policy))(x)

            hess_error = safe_norm(expert_hess - train_hess, 1e-8)
            loss = loss + config.hess_lambda * hess_error
            stats['hess_error'] = hess_error
        stats['loss'] = loss
        return loss, stats, new_batch_stats
    
    # Create the system from the config
    generator, expert, policy, init_params = make_system(config, next(rng))
    dataset_builder = functools.partial(make_dataset, generator, data_preprocess)

    # Setup the trainer
    opt_init, opt_update = optax.chain(
        # Set the parameters of Adam. Note the learning_rate is not here.
        optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        optax.additive_weight_decay(config.weight_decay),
        optax.scale_by_schedule(optax.cosine_decay_schedule(1.0,
                                config.epochs*config.train_trajs*
                                config.traj_length / config.batch_size)),
        # Put a minus sign to *minimise* the loss.
        optax.scale(-config.learning_rate))
    trainer = Trainer(opt_init, opt_update, config.epochs, config.batch_size)

    # Get the algorithm
    algo = make_algorithm(config.algorithm)

    # Run the algorithm
    logger.info(f'Running {config.algorithm}...')
    train_stats, final_params = algo(config, next(rng), expert, dataset_builder, trainer,
                        loss_fn, policy, init_params)

    # Rollout under the final policy
    logger.info('Rolling out test data...')
    generator = generator.with_key(next(rng))
    # rollout for the same key under both the expert and trained policies
    expert_test_data = generator.until(config.test_trajs)
    trained_test_data = generator.with_policy(lambda x, r: policy(final_params, x)) \
                                    .until(config.test_trajs)
    test_dataset = Dataset.zip(expert_test_data, trained_test_data).preload()
    def eval_fn(key, sample):
        expert_traj, trained_traj = sample
        final_diff = jax.vmap(lambda x: safe_norm(x, 1e-8))(expert_traj['x'] - trained_traj['x'])
        # Take the L2 norm of the diff
        delta_err = jnp.max(final_diff)
        # evaluate the expert on the trained-model rollout
        expert_us = jax.vmap(expert)(trained_traj['x'])
        # take the difference between the expert, training us
        imitation_diff = trained_traj['u'] - expert_us
        # The mean l2 error across the trajectory
        imitation_err = jnp.mean(jax.vmap(lambda x: safe_norm(x,1e-8))(imitation_diff))
        stats = {'delta_err': delta_err, 'mean_imitation_err': imitation_err }

        # If the environment supports rewards
        if 'r' in expert_traj and 'r' in trained_traj:
            stats['policy_reward'] = jnp.sum(trained_traj['r'])
            stats['expert_reward'] = jnp.sum(expert_traj['r'])
        return stats


    logger.info('Final train statistics:')
    log_info(train_stats)

    logger.info('Final test statistics:')
    test_stats = Trainer.eval(next(rng), test_dataset, eval_fn, 5)
    log_info(test_stats)

    if config.visualize:
        logger.info('Visualizing trajectory')
        generator = generator.with_key(next(rng)).with_visualize(True).with_length(config.visualize_length)
        _, expert_vis = generator.iter().next()
        _, train_vis = generator.with_policy(lambda x, r: policy(final_params, x)).iter().next()
        if 'img' in expert_vis:
            expert_imgs = np.array(expert_vis['img'])
            policy_imgs = np.array(train_vis['img'])
            fn = f'{config.environment}' if config.jac_lambda == 0 else f'{config.environment}_tasil';
            fn = fn.replace('gym/', '')
            fn = fn.replace('-v3', '')
            imageio.mimwrite(f"vis_expert_{fn}.mp4", expert_imgs, fps=10)
            imageio.mimwrite(f"vis_policy_{fn}.mp4", policy_imgs, fps=10)
        else:
            logger.info('No images to visualize')

    return {'test': test_stats, 'train': train_stats }, final_params