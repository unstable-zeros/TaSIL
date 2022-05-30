from util.rng import PRNGSequence
from util.trainer import Trainer

from loguru import logger
import functools

def bc(config, rng_key, expert, dataset_builder, trainer,
        policy_loss, policy_fn, init_params):
    def loss(keys, params, x, iteration, update_batch_stats=True):
        # Make the policy with the params
        policy = functools.partial(policy_fn, params)
        return policy_loss(keys, policy, params, x, iteration, update_batch_stats=update_batch_stats)

    rng = PRNGSequence(rng_key)
    # Make a dataset rolled out under the expert with all of the parameters
    logger.info("Generating dataset")
    dataset = dataset_builder(next(rng), expert, config.train_trajs)

    logger.info("Training")
    _, final_params = trainer.train(next(rng), dataset, loss, init_params)

    logger.info("Computing final training dataset statistics..")
    train_stats = Trainer.eval(next(rng), dataset, lambda k, x: loss(k, final_params, x, 1e10, False)[1], config.batch_size)
    return train_stats, final_params