import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
import numpy as np
import importlib
from attrdict import AttrDict
from imitation_learning import train
from util.sweep import run_all
from loguru import logger

DEFAULT_CONFIG = {
    'traj_length': 100,
    'train_trajs': 50,
    'test_trajs': 50,

    'environment': 'dummy',
    'p': 5,
    'eta': 0.3,
    'state_dim': 10,
    'activation': 'gelu',

    'gym_alternate_expert': '',
    # Tanh scaling of expert is opposite of scaling of dynamics
    # Tanh scaling of learned policy is controllable below
    'gym_scale_dynamics': False,
    'gym_scale_policy': True,
    
    'report_jac_error': True,
    'jac_lambda': 0.0,
    'jac_clip': 0.000, # gradient clipping for jac loss only

    'report_fd_error': False,
    'fd_lambda': 0.0,

    'report_hess_error': False,
    'hess_lambda': 0.0,

    'report_diff_error': False,
    'diff_lambda': 0.0,
    'diff_eps': 0.001,
    'diff_resample': True,
    'diff_augment': False,
    'diff_orthogonal': 0, # 0 uses standard normal perturbations, an integer uses n random orthogonal perturbations
    'diff_normal': 0,

    'gamma': 1.0,

    'batch_size': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-2,
    'epochs': 500,


    # Number of epochs
    'algorithm': 'bc',

    # For dagger - the mixing amount
    'beta': 0.5,

    # For dart - a noise scale selection parameter
    # if set to 0 will use T * jnp.trace(sigma_k) as in the paper
    'alpha': 0,

    'shift_iters': (1, 5, 20, 30),

    'visualize': False,
    'visualize_length': 500
}
import traceback
import warnings
import sys

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training code')
    for (k, v) in DEFAULT_CONFIG.items():
        desc = f'Default {v}' if v is not None else ''
        parser.add_argument(f'--{k}', type=type(v), help=desc)
    parser.add_argument(f'--wandb', action='store_true')
    parser.add_argument(f'--seed', default=None, type=int)
    parser.add_argument(f'--cfg', default=[], action='append')
    parser.add_argument(f'--name', default=None)
    parser.add_argument(f'--output_base', default='results')

    # An alternate way of specifying the order. If specified will
    # override jac_lambda with 1/hess_lambda with 10
    # This is to make sweeps over the tasil order easier
    parser.add_argument(f'--tasil_order', default=None, type=int)
    args = parser.parse_args()
    if args.tasil_order is not None:
        if args.tasil_order == 1:
            args.jac_lambda = 1
        elif args.tasil_order == 2:
            args.jac_lambda = 1
            args.hess_lambda = 10

    if not args.seed:
        args.seed = np.random.randint(0, 0x7fffffff)
        logger.info(f'Using randomly generated seed {args.seed} for all runs')

    config = AttrDict(DEFAULT_CONFIG)
    # Patch in the non-none elements from the arguments
    config.update({ k: v for k, v in vars(args).items() if v is not None})
    for k in {'cfg', 'output_base', 'name', 'wandb'}:
        if k in config:
            del config[k]
    configs = [config]
    # If we are doing a sweep, transform the config by the sweep module
    for transform in args.cfg:
        c = importlib.import_module(f'configs.{transform}')
        new_configs = []
        for conf in configs:
            mapped = c.transform(conf)
            if not isinstance(mapped, list):
                mapped = [mapped]
            new_configs.extend(mapped)
        configs = new_configs
    run_all(train, configs,
            args.name or '_'.join(args.cfg), args.output_base, args.wandb)