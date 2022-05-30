from util.rng import PRNGSequence
from util.trainer import Trainer
from util.dataset import Dataset

from loguru import logger
import functools

def dagger(config, rng_key, expert, dataset_builder, trainer,
        policy_loss, policy_fn, init_params):
    def loss(key, params, x, iteration, update_batch_stats=True):
        # Make the policy with the params
        policy = functools.partial(policy_fn, params)
        return policy_loss(key, policy, params, x, iteration, update_batch_stats=update_batch_stats)

    decay = config.beta
    rng = PRNGSequence(rng_key)

    trajs = config.train_trajs
    iters = list(filter(lambda x: x < trajs, config.shift_iters))
    iters.append(trajs)

    dataset = None
    trained_params = init_params
    prev_iter = 0
    beta = 1
    for i in iters:
        num_trajs = i - prev_iter
        prev_iter = i
        prev_policy = functools.partial(policy_fn, trained_params)
        mixed_policy = lambda x, r=None: beta*expert(x) + (1 - beta)*prev_policy(x)
        logger.info(f"Generating additional {num_trajs} trajectories")
        new_data = dataset_builder(next(rng), mixed_policy, num_trajs)
        # aggregate the data!
        dataset = Dataset.join(dataset, new_data) if dataset is not None else new_data
        _, trained_params = trainer.train(next(rng), dataset, loss, trained_params)
        beta = beta*decay # Decay the beta
    logger.info("Computing final training dataset statistics...")
    train_stats = Trainer.eval(next(rng), dataset, lambda k, x: loss(k, trained_params, x, 1e10, False)[1], config.batch_size)
    return train_stats, trained_params