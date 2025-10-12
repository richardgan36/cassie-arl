# Python built-in packages
import functools
from pathlib import Path

# Third-party packages
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
from absl import logging
from mujoco_playground import wrapper

from cassie_arl.rl_env.cassie_env import CassieEnv
from cassie_arl.rl_env.cassie_domain_randomizer import domain_randomize
from cassie_arl.training.ppo_callbacks import ProgressCallback, VisualizePolicyCallback


script_dir = Path(__file__).parent.resolve()
logging.set_verbosity(logging.INFO)

network_factory_params = {
    "policy_hidden_layer_sizes": (512, 256, 128),
    # "policy_obs_key": "state",
    "value_hidden_layer_sizes": (512, 256, 128),
    # "value_obs_key": "privileged_state",
    # "init_noise_std": 2.0,          # Added to increase exploration
    # "state_dependent_std": True,    # Added to increase exploration
}

ppo_training_params = {
    'action_repeat': 1,
    'batch_size': 2048,
    'clipping_epsilon': 0.2,
    'discounting': 0.97,  # TODO: Used to be 0.97. Change back?
    'entropy_cost': 0.005,  # Increased initially to encourage exploration. TODO: anneal down to 0?
    'episode_length': 1024,
    'learning_rate': 3e-4,  # Was 3e-4
    'max_grad_norm': 1.0,
    'normalize_observations': True,
    'num_envs': 4096,
    'num_evals': 22,
    'num_minibatches': 32,
    'num_resets_per_eval': 1,
    'num_timesteps': 150_000_000,
    'num_updates_per_batch': 4,
    'reward_scaling': 1.0,
    'unroll_length': 20,
    'restore_value_fn': True,
}

train_id = "parameterized_pd"  # Try to use PD control
iteration = 6
test_mode = False  # If True, run a short training for testing purposes
env = CassieEnv()

# JIT-wrapped env functions kept as local variables and passed into the visualization callback
jit_step = jax.jit(env.step)
jit_reset = jax.jit(env.reset)

# randomizer = domain_randomize
randomizer = None

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **dict(network_factory_params)
)

if test_mode:
    logging.info("\n----------\nRunning in test mode.\n----------")
    # --- Shorten training for testing ---
    ppo_training_params["num_evals"] = 2
    ppo_training_params["episode_length"] = 5
    ppo_training_params["num_envs"] = 1
    ppo_training_params["num_minibatches"] = 4
    ppo_training_params["batch_size"] = 2
    ppo_training_params["unroll_length"] = 8
    ppo_training_params["num_timesteps"] = 106

logging.info("PPO training parameters:")
logging.info(ppo_training_params)

save_ckpt_dir = script_dir / "checkpoints" / f"{train_id}" / f"iter_{iteration:02d}"
restore_ckpt_path = script_dir / "checkpoints/parameterized_pd/iter_03/000013107200"

# Instantiate callback objects
progress_cb = ProgressCallback(ppo_training_params, script_dir, train_id, iteration, save_plot=True)
viz_cb = VisualizePolicyCallback(env, jit_reset, jit_step, script_dir, train_id, iteration, run_every_n_calls=1, test_mode=test_mode)

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress_cb,
    policy_params_fn=viz_cb,
    save_checkpoint_path=save_ckpt_dir.as_posix() if not test_mode else None,
    restore_checkpoint_path=restore_ckpt_path.as_posix()
)

if "save_checkpoint_path" in train_fn.keywords and train_fn.keywords["save_checkpoint_path"] is not None:
    logging.info(f"Checkpoints will be saved to {train_fn.keywords['save_checkpoint_path']}")

# Start training
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=CassieEnv(),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)

if hasattr(progress_cb, "times") and len(progress_cb.times) >= 2:
    logging.info(f"time to train: {progress_cb.times[-1] - progress_cb.times[1]}")
else:
    logging.info("time to train: not enough progress timestamps available")

