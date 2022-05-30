from util.rng import PRNGSequence
from util.trainer import Trainer
from util.dataset import Dataset

from loguru import logger
import functools
import jax.numpy as jnp
import jax

def dart(config, rng_key, expert, dataset_builder, trainer,
        policy_loss, policy_fn, init_params):
    def loss(key, params, x, iteration, update_batch_stats=True):
        # Make the policy with the params
        policy = functools.partial(policy_fn, params)
        return policy_loss(key, policy, params, x, iteration, update_batch_stats=update_batch_stats)

    alpha = config.alpha
    rng = PRNGSequence(rng_key)

    trajs = config.train_trajs
    iters = list(filter(lambda x: x < trajs, config.shift_iters))
    iters.append(trajs)

    def noisy_policy(noise_cov, x, rng):
        expert_u = expert(x)
        # Inject gaussian noise according to the noise covariance
        return expert_u + jax.random.multivariate_normal(rng, 
                                jnp.zeros_like(expert_u), noise_cov)

    dataset = None
    alpha_cov = None
    trained_params = init_params
    prev_iter = 0
    for i in iters:
        num_trajs = i - prev_iter
        prev_iter = i
        # if we have a covariance, inject noise (first interaction just use the expert)
        # This is what the berkeley-provided DART implementation does (not mentioned in paper)
        injected_policy = functools.partial(noisy_policy, alpha_cov) if alpha_cov is not None else expert
        # Rollout data using the inected policy
        logger.info(f"Generating additional {num_trajs} trajectories")
        new_data = dataset_builder(next(rng), injected_policy, num_trajs)
        # aggregate the data!
        dataset = Dataset.join(dataset, new_data) if dataset is not None else new_data
        _, trained_params = trainer.train(next(rng), dataset, loss, trained_params)
        # Update the cov, alpha_cov parameters on the new_data
        # Since new_data is just a pytree dataset we can directly access the data
        learned_policy = functools.partial(policy_fn, trained_params)

        # We could theoretically just do this before the trainer.train but just to be consistent
        # with the original DART implementation (which seems to roll out more trajectories than expected?)
        stats_data = dataset_builder(next(rng), injected_policy, 5)
        diff = jax.vmap(learned_policy)(stats_data.data['x']) - jax.vmap(expert)(stats_data.data['x'])
        diff = jnp.expand_dims(diff, -1)
        diff_T = jnp.transpose(diff, (0, 2, 1))
        cov = jnp.mean(diff @ diff_T, 0)
        if alpha > 0:
            alpha_cov = alpha*cov / (config.traj_length * jnp.trace(cov))
        else:
            alpha_cov = cov
    logger.info("Computing final training dataset statistics...")
    train_stats = Trainer.eval(next(rng), dataset, lambda k, x: loss(k, trained_params, x, 1e10, False)[1], config.batch_size)
    return train_stats, trained_params