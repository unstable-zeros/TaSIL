import jax
import jax.numpy as jnp
from util.models import MLP
from imitation_learning.rollout import TrajectoryGenerator, \
            ArtificialSystem, ExponentialSystem, ModelPolicy, NormalGenerator, flatten_trajectory
from imitation_learning.gym import GymGenerator, make_gym_expert, make_gym_policy
import gym

# Will return the trajectory generator and expert policy
def make_system(config, rng_key):
    if config.environment == 'dummy' or config.environment == 'dummy2':
        expert_rng, policy_rng, gen_rng = jax.random.split(rng_key, 3)
        expert_model = MLP((32, 32, config.state_dim), config.activation, 
            kernel_init=jax.nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal"),
            bias_init=jax.nn.initializers.normal(0.1))
        expert_params = expert_model.init(expert_rng, jnp.zeros(config.state_dim))
        def expert(x, rng=None):
            mu = expert_model.apply(expert_params, x)
            # return mu
            return jax.nn.tanh(mu)
        generator = TrajectoryGenerator(
            traj_length=config.traj_length,
            init_gen=NormalGenerator(config.state_dim),
            dynamics=ArtificialSystem(5, 0.3, config.gamma, expert) if config.environment == 'dummy' else \
                     ExponentialSystem(0.95, 5, config.gamma, expert),
            policy=expert,
            rng_key=gen_rng
        )
        model = MLP((64, 64, 64, config.state_dim), config.activation)
        init_params = model.init(policy_rng, jnp.zeros(config.state_dim), use_running_average=False)
        def policy(vars, x, rng=None, update_batch_stats=False):
            if update_batch_stats:
                mu, bs = model.apply(vars, x, mutable=['batch_stats'], use_running_average=False)
            else:
                mu = model.apply(vars, x, use_running_average=True)
            # return (mu, bs) if update_batch_stats else mu
            mu = jax.nn.tanh(mu)
            return (mu, bs) if update_batch_stats else mu
            
        return generator, expert, policy, init_params
    elif config.environment.startswith("gym"):
        policy_rng, gen_rng = jax.random.split(rng_key, 2)

        env_name = config.environment.split("/")[1]
        env = gym.make(env_name)
        expert_name = config.gym_alternate_expert or env_name
        expert = make_gym_expert(expert_name, env, not config.gym_scale_dynamics)
        policy, init_params = make_gym_policy(policy_rng, env_name, env, config.gym_scale_policy)
        generator = GymGenerator(
            traj_length=config.traj_length,
            policy=expert,
            rng_key=rng_key,
            env=env,
            scale_dynamics=config.gym_scale_dynamics,
            vis=False
        )
        return generator, expert, policy, init_params
    else:
        raise ValueError("Unrecognized environment")

def make_dataset(generator, preprocess_pipeline,
            rng_key, policy, num_trajectories, flatten=True):
    # Switch the generator to the requested policy, rng_key
    generator = generator.with_key(rng_key)
    generator = generator.with_policy(policy)
    generator = preprocess_pipeline(generator)
    generator = generator.until(num_trajectories)
    if flatten:
        generator = generator.flat_map(flatten_trajectory)
    dataset = generator.preload()
    return dataset