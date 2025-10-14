import functools
from pathlib import Path
from typing import Optional

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.acme import running_statistics
import jax
from absl import logging
from mujoco_playground import wrapper
from orbax import checkpoint as ocp

from cassie_arl.cassie_env.cassie_env import CassieEnv, default_config
from cassie_arl.cassie_env.domain_randomization import domain_randomize
from cassie_arl.training.ppo_callbacks import ProgressCallback, VisualizePolicyCallback


script_dir = Path(__file__).parent.resolve()
logging.set_verbosity(logging.INFO)

train_id = "dr_baseline"
iteration = 1
test_mode = False

# Domain randomization baseline: no pushes, no adversary, only domain randomization
use_domain_randomization = True

network_factory_params = {
    "policy_hidden_layer_sizes": (512, 256, 128),
    "policy_obs_key": "state",  # Policy uses noisy observations
    "value_hidden_layer_sizes": (512, 256),
    "value_obs_key": "privileged_state",  # Critic uses noiseless privileged state
}

ppo_training_params = {
    'action_repeat': 1,
    'batch_size': 2048,
    'clipping_epsilon': 0.2,
    'discounting': 0.97,
    'entropy_cost': 0.005,
    'episode_length': 1024,
    'learning_rate': 3e-4,
    'max_grad_norm': 1.0,
    'normalize_observations': True,
    'num_envs': 4096,
    'num_evals': 30,
    'num_minibatches': 16,
    'num_resets_per_eval': 1,
    'num_timesteps': 200_000_000,
    'num_updates_per_batch': 4,
    'reward_scaling': 1.0,
    'unroll_length': 20,
    'restore_value_fn': True,
}


def main():
    config = default_config()
    config.push_config.enabled = False  # Disable random pushes for baseline
    
    logging.info("="*60)
    logging.info("BASELINE TRAINING: Domain Randomization Only")
    logging.info("="*60)
    logging.info("Configuration:")
    logging.info(f"  - Random pushes: DISABLED")
    logging.info(f"  - Adversary: DISABLED")
    logging.info(f"  - Domain randomization: ENABLED")
    logging.info("="*60 + "\n")
    
    env = CassieEnv(config=config, adversary_policy_fn=None)

    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **dict(network_factory_params)
    )

    if test_mode:
        logging.info("\n----------\nRunning in testing mode.\n----------")
        # --- Shorten training for testing ---
        ppo_training_params["num_evals"] = 4
        ppo_training_params["num_envs"] = 8
        ppo_training_params["learning_rate"] = 1e-6
        ppo_training_params["num_minibatches"] = 4
        ppo_training_params["batch_size"] = 2
        ppo_training_params["num_timesteps"] = 10

    logging.info("PPO training parameters:")
    logging.info(ppo_training_params)

    save_ckpt_dir = script_dir / "checkpoints" / "cassie" / f"{train_id}" / f"iter_{iteration:02d}"
    restore_ckpt_path = script_dir / "placeholder"

    # Instantiate callback objects
    progress_cb = ProgressCallback(
        ppo_training_params["num_timesteps"],
        script_dir,
        train_id,
        iteration,
        save_plot=True if not test_mode else False,
        agent_type="cassie"
    )

    viz_cb = VisualizePolicyCallback(
        env,
        jit_reset,
        jit_step,
        script_dir,
        train_id,
        iteration,
        run_every_n_calls=1,
        skip_first_n_calls=0,
        test_mode=test_mode,
        agent_type="cassie"
    )

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress_cb,
        policy_params_fn=viz_cb,
        save_checkpoint_path=save_ckpt_dir.as_posix() if not test_mode else None,
        restore_checkpoint_path=restore_ckpt_path.as_posix()
    )

    if "save_checkpoint_path" in train_fn.keywords and train_fn.keywords["save_checkpoint_path"] is not None:
        logging.info(f"Checkpoints will be saved to {train_fn.keywords['save_checkpoint_path']}")

    logging.info("\n" + "="*50 + "\nStarting Cassie baseline training\n" + "="*50 + "\n")

    randomization_fn = domain_randomize if use_domain_randomization else None

    # Start training
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=CassieEnv(config=config, adversary_policy_fn=None),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomization_fn,
    )

    if hasattr(progress_cb, "times") and len(progress_cb.times) >= 2:
        logging.info(f"Time to train: {progress_cb.times[-1] - progress_cb.times[1]}")
    else:
        logging.info("Time to train: not enough progress timestamps available")

    logging.info("\nCassie baseline training completed!")


if __name__ == "__main__":
    main()
