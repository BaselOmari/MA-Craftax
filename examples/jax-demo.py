"""
This script demonstrates how to:
- Create multiple batched Multi-Agent Craftax environments
- Vectorize environment interaction using jax.vmap
- Compile functionality using jax.jit
- Simulate N steps of interaction
"""

import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name


# ======== Configuration ========
NUM_ENVS = 64  # Number of parallel environments to simulate
N_STEPS = 32  # Number of timesteps to run in each environment


# ======== Environment Setup ========
# Create the master RNG key
master_rng = jax.random.PRNGKey(0)

# Initialize environments and reset all of them in parallel
env = make_craftax_env_from_name("Craftax-Coop-Symbolic")
master_rng, reset_rng = jax.random.split(master_rng)
reset_keys = jax.random.split(reset_rng, NUM_ENVS)

# Use `vmap` to apply reset across all environments at once
batched_reset = jax.vmap(env.reset, in_axes=(0,))
obs, states = batched_reset(reset_keys)


# ======== Rollout Execution ========
@jax.jit  # This function is JIT-compiled for performance
def step_fn(carry, _):
    """
    Single step of the rollout across all environments. This is used inside `jax.lax.scan`.
    """
    rng, states = carry

    # Create random keys for each environment for action sampling and stepping
    rng, step_rng = jax.random.split(rng)
    step_keys = jax.random.split(step_rng, NUM_ENVS)

    def sample_actions_single(rng):
        """
        Samples a random action for each agent in one environment.

        This function can be replaced with your own policy
        """
        subkeys = jax.random.split(rng, env.num_agents)
        return {
            agent: env.action_space(agent).sample(subkeys[i])
            for i, agent in enumerate(env.agents)
        }

    # Sample random actions per env
    batched_sample_actions = jax.vmap(sample_actions_single, in_axes=(0,))
    actions = batched_sample_actions(step_keys)

    # Step through all environments in parallel
    # This uses `vmap` to vectorize the step function across all environments
    batched_step = jax.vmap(env.step, in_axes=(0, 0, 0))
    obs, new_states, rewards, dones, infos = batched_step(step_keys, states, actions)

    return (rng, new_states), (obs, rewards, dones, infos)


# Run the rollout using JAX's scan (like a loop but optimized for speed and JIT)
(final_rng, final_states), (obs_seq, reward_seq, done_seq, info_seq) = jax.lax.scan(
    step_fn,  # Function to run at each time step
    (master_rng, states),  # Initial carry (rng, env states)
    None,
    length=N_STEPS,  # Number of steps to simulate
)

print(f"Rollout complete: {N_STEPS} steps across {NUM_ENVS} environments")
