import functools
from pathlib import Path

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
from absl import logging
from mujoco_playground import wrapper
from orbax import checkpoint as ocp

from cassie_arl.cassie_env.cassie_env import CassieEnv
from cassie_arl.adversary_env.adversary_env import AdversaryEnv, default_adversary_config
from cassie_arl.training.ppo_callbacks import ProgressCallback, VisualizePolicyCallback


script_dir = Path(__file__).parent.resolve()
logging.set_verbosity(logging.INFO)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Adversary training configuration
train_id = "final"
iteration = 3
test_mode = False

# Cassie checkpoint to use as frozen protagonist
cassie_checkpoint_path = script_dir / "checkpoints/cassie/active_recovery/iter_02/000136970240"

# Network architecture for adversary
network_factory_params = {
    "policy_hidden_layer_sizes": (256, 128, 64),
    "value_hidden_layer_sizes": (512, 256),
}

# PPO training parameters for adversary
ppo_training_params = {
    'action_repeat': 1,
    'batch_size': 1024,
    'clipping_epsilon': 0.2,
    'discounting': 0.97,
    'entropy_cost': 0.01,
    'episode_length': 800,
    'learning_rate': 3e-4,
    'max_grad_norm': 1.0,
    'normalize_observations': True,
    'num_envs': 1024,
    'num_evals': 20,
    'num_minibatches': 16,
    'num_resets_per_eval': 1,
    'num_timesteps': 100_000_000,
    'num_updates_per_batch': 4,
    'reward_scaling': 1.0,
    'unroll_length': 20,
    'restore_value_fn': True,
}

# ---------------------------------------------------------------------------
# Load frozen Cassie policy
# ---------------------------------------------------------------------------

def load_cassie_policy(checkpoint_path: Path):
    """
    Load a trained Cassie policy from Orbax checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        A frozen policy function (obs, rng) -> (action, extra)
    """
    logging.info(f"Loading Cassie policy from {checkpoint_path}")
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(checkpoint_path)
    temp_env = CassieEnv()  # To get observation and action sizes
    
    # Recreate the network WITH THE SAME NORMALIZATION SETTINGS used during training
    # IMPORTANT: Cassie was trained with normalize_observations=True, so we must
    # use the running_statistics.normalize function to match the trained policy
    from brax.training.acme import running_statistics
    
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256),
    )
    
    normalize_fn = running_statistics.normalize
    
    network = network_factory(
        observation_size=temp_env.observation_size,
        action_size=temp_env.action_size,
        preprocess_observations_fn=normalize_fn,
    )
    
    # Create inference function
    # The params structure from PPO training with normalize_observations=True is:
    # [normalizer_params_dict, policy_params, value_params]
    # We need to convert the normalizer_params_dict to RunningStatisticsState
    normalizer_dict, policy_params, value_params = params
    normalizer_params = running_statistics.RunningStatisticsState(
        mean=normalizer_dict['mean'],
        std=normalizer_dict['std'],
        count=normalizer_dict['count'],
        summed_variance=normalizer_dict['summed_variance']
    )
    
    # Reconstruct params with the proper structure
    params_with_normalizer = [normalizer_params, policy_params, value_params]
    
    make_policy = ppo_networks.make_inference_fn(network)
    policy_fn = make_policy(params_with_normalizer, deterministic=True)
    
    logging.info("Successfully loaded Cassie policy with observation normalization")
    
    return policy_fn


# --------------------------------------------------------------------------
# Main training
# --------------------------------------------------------------------------

def main():
    if test_mode:
        logging.info("\n----------\nRunning in testing mode.\n----------")
        # Shorten training for testing
        ppo_training_params["num_evals"] = 3
        ppo_training_params["num_envs"] = 32
        ppo_training_params["learning_rate"] = 3e-4  # Keep normal LR for testing
        ppo_training_params["num_minibatches"] = 4
        ppo_training_params["batch_size"] = 128
        ppo_training_params["num_timesteps"] = 160  # At least 2 evals worth
    
    logging.info("PPO training parameters:")
    logging.info(ppo_training_params)

    # Load frozen Cassie policy
    cassie_policy_fn = load_cassie_policy(cassie_checkpoint_path)
    
    # Create adversary environment
    adversary_config = default_adversary_config()
    env = AdversaryEnv(
        cassie_policy_fn=cassie_policy_fn,
        config=adversary_config,
    )
    
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)
    
    # Network factory for adversary
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **dict(network_factory_params)
    )
    
    save_ckpt_dir = script_dir / "checkpoints" / "adversary" / f"{train_id}" / f"iter_{iteration:02d}"
    
    # Progress callback for plotting rewards and logging progress
    progress_cb = ProgressCallback(
        ppo_training_params["num_timesteps"],
        script_dir,
        train_id,
        iteration,
        save_plot=True if not test_mode else False,
        agent_type="adversary"
    )
    
    # Visualization callback for adversary
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
        agent_type="adversary",
    )
    
    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=progress_cb,
        policy_params_fn=viz_cb,
        save_checkpoint_path=save_ckpt_dir.as_posix() if not test_mode else None,
    )
    
    if "save_checkpoint_path" in train_fn.keywords and train_fn.keywords["save_checkpoint_path"] is not None:
        logging.info(f"Checkpoints will be saved to {train_fn.keywords['save_checkpoint_path']}")
    
    # Start training
    logging.info("\n" + "="*50 + "\nStarting adversary training\n" + "="*50 + "\n")
    
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=AdversaryEnv(
            cassie_policy_fn=cassie_policy_fn,
            config=adversary_config,
        ),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    
    if hasattr(progress_cb, "times") and len(progress_cb.times) >= 2:
        logging.info(f"Time to train: {progress_cb.times[-1] - progress_cb.times[1]}")
    else:
        logging.info("Time to train: not enough progress timestamps available")
    
    logging.info("\nAdversary training completed!")


if __name__ == "__main__":
    main()
