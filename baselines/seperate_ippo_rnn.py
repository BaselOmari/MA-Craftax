"""
Code is adapted from the IPPO RNN implementation of JaxMARL (https://github.com/FLAIROx/JaxMARL/tree/main) 
Credit goes to the original authors: Rutherford et al.

Modified to use SEPARATE network parameters per agent (no parameter sharing).
Each agent has its own ActorCriticRNN with independent parameters.
Gradient clipping is done PER-AGENT to avoid coupling through global norm computation.
"""

# ===========================
# Imports and Configuration
# ===========================
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import functools
import yaml
from typing import Sequence, NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import optax
import distrax

import wandb

import imageio

from jaxmarl.wrappers.baselines import LogWrapper
from craftax.craftax_env import make_craftax_env_from_name
from craftax.environment_base.wrappers import VideoPlotWrapper
from craftax.custom_rendering.base_rendering import load_rendering_resources
from craftax.custom_rendering.ego_rendering import render_ego_perspective
from craftax.custom_rendering.full_map_rendering import render_full_map

# ===========================
# Model Definitions
# ===========================
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

# ===========================
# Data Structures and Utilities
# ===========================
class Transition(NamedTuple):
    """Full transition including info for logging."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class TrainBatch(NamedTuple):
    """Batch for PPO update (without info to avoid minibatch issues)."""
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

def batchify(x: dict, agent_list):
    """Stack agent observations, preserving agent dimension.
    
    Returns shape: (num_agents, num_envs, obs_dim)
    """
    return jnp.stack([x[a] for a in agent_list], axis=0)

def unbatchify(x: jnp.ndarray, agent_list):
    """Convert stacked array back to agent dict.
    
    Input shape: (num_agents, num_envs, ...) or (num_agents, num_envs)
    """
    return {a: x[i] for i, a in enumerate(agent_list)}

# ===========================
# Training Function
# ===========================
def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # Note: In separate IPPO, minibatching is done over NUM_ENVS per agent
    # Each minibatch has shape (num_steps, num_agents, num_envs // NUM_MINIBATCHES, ...)
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] // config["NUM_MINIBATCHES"]

    # Load rendering resources BEFORE wrapping (need base env's static_env_params)
    _video_env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    _video_pixel_size = config.get("VIDEO_PIXEL_SIZE", 16)
    _video_static_params = env.static_env_params
    _video_rendering_res = load_rendering_resources(_video_env_name, pixel_size_preference=_video_pixel_size)
    _video_textures = _video_rendering_res["TEXTURES"]
    _video_player_textures = _video_rendering_res["load_player_specific_textures"](
        _video_textures[_video_pixel_size], _video_static_params.player_count
    )

    env = LogWrapper(env)
    env = VideoPlotWrapper(env, './output/', 256, False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # Per-agent gradient clipping to avoid coupling agents through global norm
    def per_agent_clip_by_global_norm(max_norm):
        """Clip gradients per agent independently, not across all agents."""
        def init_fn(params):
            del params
            return optax.EmptyState()
        
        def update_fn(updates, state, params=None):
            del params
            # updates has shape (num_agents, ...) for each leaf
            # We need to clip each agent's gradients independently
            
            def clip_single_agent(agent_grads):
                # Compute norm for this agent only
                leaves = jax.tree_util.tree_leaves(agent_grads)
                sum_of_squares = sum(jnp.sum(jnp.square(x)) for x in leaves)
                norm = jnp.sqrt(sum_of_squares)
                # Clip
                scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
                return jax.tree_util.tree_map(lambda x: x * scale, agent_grads)
            
            # Vmap over the agent dimension (axis 0 of each leaf)
            clipped_updates = jax.vmap(clip_single_agent)(updates)
            return clipped_updates, state
        
        return optax.GradientTransformation(init_fn, update_fn)

    def train(rng):
        # INIT NETWORK - separate params per agent
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        
        # Initialize separate params for each agent using vmap
        agent_rngs = jax.random.split(_rng, env.num_agents)
        
        def init_single_agent(agent_rng):
            return network.init(agent_rng, init_hstate, init_x)
        
        # Stacked network variables: leading dim is num_agents
        stacked_network_variables = jax.vmap(init_single_agent)(agent_rngs)
        # Extract only params (network.init returns {"params": ...})
        stacked_network_params = stacked_network_variables["params"]
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                per_agent_clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                per_agent_clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=stacked_network_params,  # (num_agents, ...) - only params, not full variables
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        # Hidden state shape: (num_agents, num_envs, hidden_dim)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        init_hstate = jnp.tile(init_hstate[np.newaxis, :, :], (env.num_agents, 1, 1))
        # Initial done flags as dict (will be converted to array in _env_step)
        # Must include "__all__" key to match structure returned by env.step
        init_done = {a: jnp.zeros((config["NUM_ENVS"],), dtype=bool) for a in env.agents}
        init_done["__all__"] = jnp.zeros((config["NUM_ENVS"],), dtype=bool)

        # TRAIN LOOP
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            # obs_batch shape: (num_agents, num_envs, obs_dim)
            obs_batch = batchify(last_obs, env.agents)
            # done_batch shape: (num_agents, num_envs)
            # last_done is a dict from env, convert to array
            done_batch_in = batchify(last_done, env.agents)
            
            # Forward pass for each agent with their own params
            # ac_in: (1, num_envs, obs_dim), (1, num_envs)
            # hstate: (num_agents, num_envs, hidden_dim)
            def forward_single_agent(params, hs, obs, done):
                ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
                return network.apply({"params": params}, hs, ac_in)
            
            hstate, pi, value = jax.vmap(forward_single_agent)(
                train_state.params,  # (num_agents, ...)
                hstate,              # (num_agents, num_envs, hidden_dim)
                obs_batch,           # (num_agents, num_envs, obs_dim)
                done_batch_in,       # (num_agents, num_envs)
            )
            # pi.logits shape: (num_agents, 1, num_envs, action_dim)
            # value shape: (num_agents, 1, num_envs)
            
            # Sample actions - distrax is batch-aware, sample directly
            # pi.logits: (num_agents, 1, num_envs, action_dim)
            action = pi.sample(seed=_rng)  # (num_agents, 1, num_envs)
            log_prob = pi.log_prob(action)  # (num_agents, 1, num_envs)
            
            action = action.squeeze(axis=1)      # (num_agents, num_envs)
            log_prob = log_prob.squeeze(axis=1)  # (num_agents, num_envs)
            value = value.squeeze(axis=1)        # (num_agents, num_envs)
            
            env_act = unbatchify(action, env.agents)
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(rng_step, env_state, env_act)
            
            done_batch = batchify(done, env.agents)  # (num_agents, num_envs)
            reward_batch = batchify(reward, env.agents)  # (num_agents, num_envs)
            
            transition = Transition(
                jnp.tile(done["__all__"][np.newaxis, :], (env.num_agents, 1)),  # (num_agents, num_envs)
                done_batch_in,   # (num_agents, num_envs)
                action,          # (num_agents, num_envs)
                value,           # (num_agents, num_envs)
                reward_batch,    # (num_agents, num_envs)
                log_prob,        # (num_agents, num_envs)
                obs_batch,       # (num_agents, num_envs, obs_dim)
                info,
            )

            # Add hstate and other non-env metrics to info so they can be logged
            info['action'] = action           # (num_agents, num_envs)
            info['done'] = done_batch         # (num_agents, num_envs)
            info['value'] = value             # (num_agents, num_envs)
            info['hidden_state'] = hstate     # (num_agents, num_envs, hidden_dim)
            # pi.entropy() returns (num_agents, 1, num_envs) - squeeze axis 1
            info['entropy'] = pi.entropy().squeeze(1)  # (num_agents, num_envs)
            info['log_prob'] = log_prob       # (num_agents, num_envs)

            # Keep done as dict for next iteration (env returns dict)
            runner_state = (train_state, env_state, obsv, done, hstate, rng)
            return runner_state, transition

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            # Save initial hidden state BEFORE rollout for PPO rerun
            initial_hstate = runner_state[4]  # hstate before rollout
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents)
            last_done_batch = batchify(last_done, env.agents)  # last_done is dict from env
            
            def forward_single_agent(params, hs, obs, done):
                ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
                return network.apply({"params": params}, hs, ac_in)
            
            _, _, last_val = jax.vmap(forward_single_agent)(
                train_state.params, hstate, last_obs_batch, last_done_batch
            )
            last_val = last_val.squeeze()  # (num_agents, num_envs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # Extract TrainBatch without info for minibatching
            train_batch = TrainBatch(
                global_done=traj_batch.global_done,
                done=traj_batch.done,
                action=traj_batch.action,
                value=traj_batch.value,
                reward=traj_batch.reward,
                log_prob=traj_batch.log_prob,
                obs=traj_batch.obs,
            )
            # Keep info separate for logging
            traj_info = traj_batch.info

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, train_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, train_batch, gae, targets):
                        # RERUN NETWORK for each agent
                        # init_hstate: (num_agents, num_envs_minibatch, hidden_dim)
                        # traj_batch.obs: (num_steps, num_agents, num_envs_minibatch, obs_dim)
                        
                        def forward_single_agent(p, hs, obs, done):
                            # obs: (num_steps, num_envs_minibatch, obs_dim)
                            # done: (num_steps, num_envs_minibatch)
                            return network.apply({"params": p}, hs, (obs, done))
                        
                        # Transpose train_batch for per-agent processing
                        obs_per_agent = jnp.transpose(train_batch.obs, (1, 0, 2, 3))  # (num_agents, num_steps, num_envs, obs_dim)
                        done_per_agent = jnp.transpose(train_batch.done, (1, 0, 2))   # (num_agents, num_steps, num_envs)
                        action_per_agent = jnp.transpose(train_batch.action, (1, 0, 2))  # (num_agents, num_steps, num_envs)
                        
                        _, pi, value = jax.vmap(forward_single_agent)(
                            params,          # (num_agents, ...)
                            init_hstate,     # (num_agents, num_envs_minibatch, hidden_dim)
                            obs_per_agent,   # (num_agents, num_steps, num_envs_minibatch, obs_dim)
                            done_per_agent,  # (num_agents, num_steps, num_envs_minibatch)
                        )
                        # pi.logits: (num_agents, num_steps, num_envs_minibatch, action_dim)
                        # value: (num_agents, num_steps, num_envs_minibatch)
                        
                        # Use distrax batch operations directly (no vmap over distribution objects)
                        log_prob = pi.log_prob(action_per_agent)
                        # log_prob: (num_agents, num_steps, num_envs_minibatch)
                        
                        # Transpose back to (num_steps, num_agents, num_envs_minibatch)
                        log_prob = jnp.transpose(log_prob, (1, 0, 2))
                        value = jnp.transpose(value, (1, 0, 2))
                        
                        # CALCULATE VALUE LOSS
                        # Shape: (num_steps, num_agents, num_envs_minibatch)
                        value_pred_clipped = train_batch.value + (
                            value - train_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss_per_elem = 0.5 * jnp.maximum(value_losses, value_losses_clipped)
                        # Per-agent value loss: mean over time (0) and envs (2), keep agents (1)
                        value_loss_per_agent = value_loss_per_elem.mean(axis=(0, 2))  # (num_agents,)
                        value_loss = value_loss_per_agent.mean()  # scalar for gradient

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - train_batch.log_prob
                        ratio = jnp.exp(logratio)
                        # Normalize advantages PER AGENT (no coupling between agents)
                        # gae shape: (num_steps, num_agents, num_envs_minibatch)
                        # Normalize over time (axis 0) and envs (axis 2), independently for each agent
                        gae_mean = gae.mean(axis=(0, 2), keepdims=True)
                        gae_std = gae.std(axis=(0, 2), keepdims=True)
                        gae = (gae - gae_mean) / (gae_std + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor_per_elem = -jnp.minimum(loss_actor1, loss_actor2)
                        # Per-agent actor loss: mean over time (0) and envs (2), keep agents (1)
                        loss_actor_per_agent = loss_actor_per_elem.mean(axis=(0, 2))  # (num_agents,)
                        loss_actor = loss_actor_per_agent.mean()  # scalar for gradient
                        
                        # Entropy: use distrax directly (batch-aware)
                        # pi.entropy() returns (num_agents, num_steps, num_envs_minibatch)
                        entropy_per_elem = pi.entropy()  # (num_agents, num_steps, num_envs)
                        entropy_per_agent = entropy_per_elem.mean(axis=(1, 2))  # (num_agents,)
                        entropy = entropy_per_agent.mean()  # scalar for gradient

                        # debug - per agent
                        approx_kl_per_agent = ((ratio - 1) - logratio).mean(axis=(0, 2))  # (num_agents,)
                        clip_frac_per_agent = (jnp.abs(ratio - 1) > config["CLIP_EPS"]).mean(axis=(0, 2))  # (num_agents,)
                        approx_kl = approx_kl_per_agent.mean()
                        clip_frac = clip_frac_per_agent.mean()

                        total_loss_per_agent = (
                            loss_actor_per_agent
                            + config["VF_COEF"] * value_loss_per_agent
                            - config["ENT_COEF"] * entropy_per_agent
                        )  # (num_agents,)
                        total_loss = total_loss_per_agent.mean()  # scalar for gradient
                        
                        # Return both scalar losses (for gradient) and per-agent losses (for logging)
                        return total_loss, (
                            value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac,
                            total_loss_per_agent, value_loss_per_agent, loss_actor_per_agent, entropy_per_agent
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, train_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    train_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # Prepare batch for minibatching
                # init_hstate: (num_agents, num_envs, hidden_dim)
                # train_batch shapes: (num_steps, num_agents, num_envs, ...)
                # advantages/targets: (num_steps, num_agents, num_envs)
                
                # Permute over num_envs dimension
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                # Shuffle init_hstate: (num_agents, num_envs, hidden_dim) -> axis 1
                init_hstate_shuffled = jnp.take(init_hstate, permutation, axis=1)
                
                # Shuffle train_batch components: (num_steps, num_agents, num_envs, ...) -> axis 2
                def shuffle_batch(x):
                    return jnp.take(x, permutation, axis=2)
                
                train_batch_shuffled = TrainBatch(
                    global_done=shuffle_batch(train_batch.global_done),
                    done=shuffle_batch(train_batch.done),
                    action=shuffle_batch(train_batch.action),
                    value=shuffle_batch(train_batch.value),
                    reward=shuffle_batch(train_batch.reward),
                    log_prob=shuffle_batch(train_batch.log_prob),
                    obs=shuffle_batch(train_batch.obs),
                )
                
                # Shuffle advantages/targets: (num_steps, num_agents, num_envs) -> axis 2
                advantages_shuffled = jnp.take(advantages, permutation, axis=2)
                targets_shuffled = jnp.take(targets, permutation, axis=2)
                
                # Create minibatches
                def minibatch_hstate(x):
                    # x: (num_agents, num_envs, hidden_dim)
                    # -> (num_minibatches, num_agents, minibatch_size, hidden_dim)
                    num_agents, num_envs, hidden_dim = x.shape
                    minibatch_size = num_envs // config["NUM_MINIBATCHES"]
                    return x.reshape(num_agents, config["NUM_MINIBATCHES"], minibatch_size, hidden_dim).swapaxes(0, 1)
                
                def minibatch_array(x):
                    # x: (num_steps, num_agents, num_envs, ...) 
                    # -> (num_minibatches, num_steps, num_agents, minibatch_size, ...)
                    shape = list(x.shape)
                    num_steps, num_agents, num_envs = shape[:3]
                    rest = shape[3:]
                    minibatch_size = num_envs // config["NUM_MINIBATCHES"]
                    new_shape = [num_steps, num_agents, config["NUM_MINIBATCHES"], minibatch_size] + rest
                    reshaped = x.reshape(new_shape)
                    # Move minibatch axis to front
                    return jnp.moveaxis(reshaped, 2, 0)
                
                init_hstate_mb = minibatch_hstate(init_hstate_shuffled)
                
                train_batch_mb = TrainBatch(
                    global_done=minibatch_array(train_batch_shuffled.global_done),
                    done=minibatch_array(train_batch_shuffled.done),
                    action=minibatch_array(train_batch_shuffled.action),
                    value=minibatch_array(train_batch_shuffled.value),
                    reward=minibatch_array(train_batch_shuffled.reward),
                    log_prob=minibatch_array(train_batch_shuffled.log_prob),
                    obs=minibatch_array(train_batch_shuffled.obs),
                )
                
                advantages_mb = minibatch_array(advantages_shuffled)
                targets_mb = minibatch_array(targets_shuffled)
                
                minibatches = (init_hstate_mb, train_batch_mb, advantages_mb, targets_mb)

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    train_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                train_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            
            # traj_info is a FrozenDict from LogWrapper - create new dict to avoid mutation issues
            # ratio_0: get before mean reduction (like original)
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            
            # Per-agent losses are now returned directly from loss_fn
            # loss_info[1][6:10] are the per-agent values: total, value, actor, entropy
            # Shape after scan: (num_epochs, num_minibatches, num_agents)
            # Mean over epochs and minibatches to get (num_agents,)
            total_loss_per_agent = loss_info[1][6].mean(axis=(0, 1))    # (num_agents,)
            value_loss_per_agent = loss_info[1][7].mean(axis=(0, 1))    # (num_agents,)
            actor_loss_per_agent = loss_info[1][8].mean(axis=(0, 1))    # (num_agents,)
            entropy_per_agent = loss_info[1][9].mean(axis=(0, 1))       # (num_agents,)
            
            # Global mean for backward compatibility
            loss_info_mean = jax.tree.map(lambda x: x.mean(), loss_info)
            
            # Create new metric dict (don't mutate FrozenDict from LogWrapper)
            metric = {
                **dict(traj_info),  # Convert FrozenDict to regular dict
                "update_steps": update_steps,
                "loss": {
                    "total_loss": loss_info_mean[0],
                    "value_loss": loss_info_mean[1][0],
                    "actor_loss": loss_info_mean[1][1],
                    "entropy": loss_info_mean[1][2],
                    "ratio": loss_info_mean[1][3],
                    "ratio_0": ratio_0,
                    "approx_kl": loss_info_mean[1][4],
                    "clip_frac": loss_info_mean[1][5],
                },
                "loss_per_agent": {
                    "total_loss": total_loss_per_agent,      # (num_agents,)
                    "value_loss": value_loss_per_agent,      # (num_agents,)
                    "actor_loss": actor_loss_per_agent,      # (num_agents,)
                    "entropy": entropy_per_agent,            # (num_agents,)
                },
            }

            rng = update_state[-1]

            def callback(metrics, actor_state: TrainState, step):
                env_step = (
                    metrics["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                )
                to_log = {
                    "env_step": env_step,
                    **metrics["loss"],
                }
                
                # Log per-agent losses (use np.asarray().item() for safe conversion)
                num_agents = metrics["loss_per_agent"]["total_loss"].shape[0]
                for i in range(num_agents):
                    to_log[f"agent_{i}/total_loss"] = np.asarray(metrics["loss_per_agent"]["total_loss"][i]).item()
                    to_log[f"agent_{i}/value_loss"] = np.asarray(metrics["loss_per_agent"]["value_loss"][i]).item()
                    to_log[f"agent_{i}/actor_loss"] = np.asarray(metrics["loss_per_agent"]["actor_loss"][i]).item()
                    to_log[f"agent_{i}/entropy"] = np.asarray(metrics["loss_per_agent"]["entropy"][i]).item()
                
                if metrics["returned_episode"].any():
                    # Log aggregated achievements (mean across all agents) - thicker line with error bars
                    to_log.update(jax.tree.map(
                        lambda x: x[metrics["returned_episode"]].mean(),
                        metrics["user_info"]
                    ))
                    # Log per-agent achievements
                    # info shape from LogWrapper: (num_steps, num_envs, num_agents)
                    num_agents = metrics["returned_episode"].shape[2]
                    
                    # Define team assignments: Team A (even indices), Team B (odd indices)
                    team_a_agents = list(range(0, num_agents, 2))  # [0, 2, 4, ...]
                    team_b_agents = list(range(1, num_agents, 2))  # [1, 3, 5, ...]
                    
                    # Collect team-level metrics
                    team_a_returns = []
                    team_b_returns = []
                    team_a_lengths = []
                    team_b_lengths = []
                    team_a_achievements = {key: [] for key in metrics["user_info"].keys()}
                    team_b_achievements = {key: [] for key in metrics["user_info"].keys()}
                    
                    for agent_idx in range(num_agents):
                        # Get mask for this agent's returned episodes
                        agent_mask = metrics["returned_episode"][:, :, agent_idx]  # (num_steps, num_envs)
                        if agent_mask.any():
                            # Log per-agent achievements
                            for key, value in metrics["user_info"].items():
                                # value shape: (num_steps, num_envs, num_agents)
                                agent_value = value[:, :, agent_idx]  # (num_steps, num_envs)
                                agent_mean = agent_value[agent_mask].mean()
                                # Log as "agent_0/Achievements/collect_wood" - organized by agent
                                to_log[f"agent_{agent_idx}/{key}"] = np.asarray(agent_mean).item()
                                
                                # Collect for team aggregation
                                if agent_idx in team_a_agents:
                                    team_a_achievements[key].append(np.asarray(agent_mean).item())
                                else:
                                    team_b_achievements[key].append(np.asarray(agent_mean).item())
                            
                            # Log per-agent episode returns and lengths
                            agent_returns = metrics["returned_episode_returns"][:, :, agent_idx][agent_mask].mean()
                            to_log[f"agent_{agent_idx}/episode_returns"] = np.asarray(agent_returns).item()
                            agent_lengths = metrics["returned_episode_lengths"][:, :, agent_idx][agent_mask].mean()
                            to_log[f"agent_{agent_idx}/episode_lengths"] = np.asarray(agent_lengths).item()
                            
                            # Collect for team aggregation
                            if agent_idx in team_a_agents:
                                team_a_returns.append(np.asarray(agent_returns).item())
                                team_a_lengths.append(np.asarray(agent_lengths).item())
                            else:
                                team_b_returns.append(np.asarray(agent_returns).item())
                                team_b_lengths.append(np.asarray(agent_lengths).item())
                    
                    # Log team-aggregated metrics (mean over team members)
                    if team_a_returns:
                        to_log["team_a/episode_returns"] = np.mean(team_a_returns)
                        to_log["team_a/episode_lengths"] = np.mean(team_a_lengths)
                        for key, values in team_a_achievements.items():
                            if values:
                                to_log[f"team_a/{key}"] = np.mean(values)
                    
                    if team_b_returns:
                        to_log["team_b/episode_returns"] = np.mean(team_b_returns)
                        to_log["team_b/episode_lengths"] = np.mean(team_b_lengths)
                        for key, values in team_b_achievements.items():
                            if values:
                                to_log[f"team_b/{key}"] = np.mean(values)
                    
                    # Log team-specific trade and combat metrics
                    # Note: same_subclass_trades tracks trades within teams (Team A with Team A, Team B with Team B)
                    # Since trades are team-restricted, this is the total valid trades
                    if "Trade/same_subclass_trades" in metrics["user_info"]:
                        trade_value = metrics["user_info"]["Trade/same_subclass_trades"]
                        # trade_value is (num_steps, num_envs, num_agents) - it's broadcast same value for all agents
                        # Just take mean over returned episodes
                        if metrics["returned_episode"].any():
                            trade_mean = trade_value[metrics["returned_episode"]].mean()
                            to_log["team_trades/total_same_team_trades"] = np.asarray(trade_mean).item()
                    
                    # Log diff_subclass_trades (should be 0 since trades are blocked between teams, but log for verification)
                    if "Trade/diff_subclass_trades" in metrics["user_info"]:
                        diff_trade_value = metrics["user_info"]["Trade/diff_subclass_trades"]
                        if metrics["returned_episode"].any():
                            diff_trade_mean = diff_trade_value[metrics["returned_episode"]].mean()
                            to_log["team_trades/blocked_cross_team_trades"] = np.asarray(diff_trade_mean).item()
                    
                    # Note: Combat/team_a_kills and Combat/team_b_kills are already logged via the general user_info loop above
                    
                    to_log["episode_lengths"] = metrics["returned_episode_lengths"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                    to_log["episode_returns"] = metrics["returned_episode_returns"][:, :, 0][
                        metrics["returned_episode"][:, :, 0]
                    ].mean()
                            
                print(to_log)
                wandb.log(to_log, step=metrics["update_steps"])

            jax.experimental.io_callback(callback, None, metric, train_state, update_steps)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), metric


        # Do one "step" of logging, writing the result to a file.
        # Several steps can be run in series using --logging_steps_per_viz to do long rollouts without hitting memory limits
        def _logging_step(carry, unused, logging_threads, update_step):
            runner_state, episode_count = carry
            # Visualization rollouts
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["LOGGING_STEPS_PER_CALL"],
            )

            # Finally, log data associated with the visualization runs

            hidden_states = traj_batch.info['hidden_state']
            # In seperate_ippo_rnn, hidden_states already has shape (T, num_agents, NUM_ENVS, hidden_dim)
            # No reshape needed - it's already in the correct format
            # Null this for memory savings
            traj_batch.info['hidden_state'] = None

            # Compute a pseudo episode_id from cumulative done flags
            # done shape: (T, num_agents, NUM_ENVS)  (network output field)
            # episode_count shape: (num_agents, NUM_ENVS) — carried across logging steps
            # Shift by 1 so the done step itself still belongs to the old episode
            done_shifted = jnp.concatenate([
                jnp.zeros((1,) + traj_batch.info['done'].shape[1:]),
                traj_batch.info['done'][:-1]
            ], axis=0)
            local_episode_id = jnp.cumsum(done_shifted, axis=0)  # (T, num_agents, NUM_ENVS)
            traj_batch.info['episode_id'] = (episode_count[None, :, :] + local_episode_id).astype(jnp.float32)
            # Update episode_count for next logging step: add total dones in this chunk
            episode_count = episode_count + traj_batch.info['done'].sum(axis=0).astype(episode_count.dtype)

            # Add new logging fields here
            fields_to_log = ['health', 'food', 'drink', 'energy', 'done', 'is_sleeping', 'is_resting',
                             'player_position_x',
                             'player_position_y', 'recover', 'hunger', 'thirst', 'fatigue', 'light_level',
                             'dist_to_melee_l1',
                             'melee_on_screen', 'dist_to_passive_l1', 'passive_on_screen', 'dist_to_ranged_l1',
                             'ranged_on_screen', 'num_melee_nearby', 'num_passives_nearby', 'num_ranged_nearby',
                             #'delta', 'pred_delta',
                             'num_monsters_killed',
                             'has_sword', 'has_pick', 'held_iron', 'value',
                             'entropy', 'log_prob', 'episode_id',
                            ]

            # Callback function for logging hidden states
            def write_rnn_hstate(hstate, scalars, increment=0, agent_n=0):

                header_field_names = ['health', 'food', 'drink', 'energy', 'done', 'is_sleeping', 'is_resting',
                                      'player_position_x',
                                      'player_position_y', 'recover', 'hunger', 'thirst', 'fatigue', 'light_level',
                                      'dist_to_melee_l1',
                                      'melee_on_screen', 'dist_to_passive_l1', 'passive_on_screen', 'dist_to_ranged_l1',
                                      'ranged_on_screen', 'num_melee_nearby', 'num_passives_nearby',
                                      'num_ranged_nearby',
                                      #'delta_x', 'delta_y', 'pred_delta_x', 'pred_delta_y',
                                      'num_monsters_killed',
                                      'has_sword',
                                      'has_pick', 'held_iron', 'value', 'entropy', 'log_prob', 'episode_id',
                                        ]

                run_out_path = os.path.join('./', wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
                # Assemble header for the scalar file(s)
                scalar_file_header = 'action'
                for key in header_field_names:
                    scalar_file_header += ',' + key

                # We save to temp files and then append to the target file since numpy apparently cannot write files in append mode for some reason
                for i in range(logging_threads):
                    out_filename_hstates = os.path.join(run_out_path, 'hstates_{}_{}_{}.csv'.format(increment, agent_n, i))
                    temp_filename = os.path.join(run_out_path, 'temp.csv')
                    np.savetxt(temp_filename,
                               hstate[:, i, :], delimiter=',')
                    temp_file = open(temp_filename, 'r')
                    out_file_hstates = open(out_filename_hstates, 'a+')
                    out_file_hstates.write(temp_file.read())
                    out_file_hstates.close()
                    temp_file.close()
                    # Then do the same thing for the scalars
                    out_filename_scalars = os.path.join(run_out_path, 'scalars_{}_{}_{}.csv'.format(increment, agent_n, i))
                    np.savetxt(temp_filename,
                               scalars[:, i, :], delimiter=',', fmt='%f',
                               header=scalar_file_header
                               )
                    temp_file = open(temp_filename, 'r')
                    out_file_scalars = open(out_filename_scalars, 'a+')
                    out_file_scalars.write(temp_file.read())
                    temp_file.close()
                    out_file_scalars.close()
                    print('Writing log file', out_filename_hstates)

            # Add the specified field to the logging array
            # In seperate_ippo_rnn:
            # - Network outputs (action, done, value, entropy, log_prob) have shape (T, num_agents, NUM_ENVS)
            # - Environment fields (health, food, etc.) have shape (T, NUM_ENVS, num_agents)
            network_output_fields = {'value', 'entropy', 'log_prob', 'done', 'action', 'episode_id'}
            
            def add_field_to_log_array(info_dict, log_array, field_key, agent_to_log):
                field_value = info_dict[field_key]
                # Select the current agent if this is a per-agent field
                if len(field_value.shape) == 3:
                    if field_key in network_output_fields:
                        # Network outputs: shape (T, num_agents, NUM_ENVS)
                        field_value = field_value[:, agent_to_log, :]
                    else:
                        # Environment fields: shape (T, NUM_ENVS, num_agents)
                        field_value = field_value[:, :, agent_to_log]
                new_shape = field_value.shape + (1,)
                field_value = field_value.reshape(new_shape)

                log_array = jnp.concatenate([log_array, field_value], axis=2)

                return log_array

            # Assemble logging variable array
            # In seperate_ippo_rnn, network outputs already have shape (T, num_agents, NUM_ENVS)
            # No reshape needed - shapes are already correct
            for agent_n in range(env.num_agents):
                # Network outputs have shape (T, num_agents, NUM_ENVS) - extract agent_n -> (T, NUM_ENVS)
                log_array = traj_batch.info['action'][:, agent_n, :].reshape((traj_batch.info['action'].shape[0], config['NUM_ENVS'], 1))
                # Yes this is a for loop in the JAX code but this stuff was getting done in serial before anyway and it's cheap operations
                for field_to_log in fields_to_log:
                    log_array = add_field_to_log_array(traj_batch.info, log_array, field_to_log, agent_n)

                # Extract hidden states only for this agent: (T, NUM_ENVS, hidden_dim)
                agent_hidden_states = hidden_states[:, agent_n, :, :]
                jax.debug.callback(write_rnn_hstate, agent_hidden_states, log_array, update_step, agent_n)

            return (runner_state, episode_count), None

            # Func to interleave update steps and plotting

        # ===========================
        # Video Rollout Step
        # ===========================
        def _video_env_step(runner_state, unused):
            """One env step that also renders ego + full_map frames for video."""
            runner_state, transition = _env_step(runner_state, unused)
            # Unwrap env_state: runner_state[1] is vmapped LogEnvState
            # LogEnvState.env_state -> base craftax EnvState
            # We take env 0 from the vmap batch for video
            vmapped_env_state = runner_state[1]
            craftax_state = jax.tree_util.tree_map(lambda x: x[0], vmapped_env_state.env_state)

            # Render ego-perspective: (num_agents, H, W, 3) float32 [0, 255]
            ego_frames = render_ego_perspective(
                craftax_state, _video_pixel_size, _video_static_params,
                _video_player_textures, _video_env_name
            )

            # Render full map: (map_H, map_W, 3) float32 [0, 255]
            full_map_frame = render_full_map(
                craftax_state, _video_static_params,
                _video_textures[_video_pixel_size], _video_player_textures,
                _video_pixel_size, env_name=_video_env_name
            )

            return runner_state, (ego_frames, full_map_frame)

        def _update_plot(runner_state, unused):
            # First, do iterations of logging
            state, update_steps = runner_state
            # episode_count tracks cumulative episode IDs across logging steps: (num_agents, NUM_ENVS)
            episode_count = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=jnp.int32)
            (state, episode_count), empty = jax.lax.scan(
                functools.partial(_logging_step, logging_threads=config["LOGGING_THREADS"], update_step=update_steps),
                (state, episode_count), None,
                config["LOGGING_NUM_CALLS"],
            )

            # ===========================
            # Video Rollout (fresh episode so we see the overworld spawn)
            # ===========================
            video_length = config.get("VIDEO_LENGTH", 200)

            # Unpack training state
            train_state_v, env_state_v, obsv_v, done_v, hstate_v, rng_v = state

            # Fork RNG: one for video, one to continue training
            rng_video, rng_continue = jax.random.split(rng_v)

            # Reset envs to get a fresh episode (overworld, level 0)
            video_reset_rngs = jax.random.split(rng_video, config["NUM_ENVS"])
            video_obsv, video_env_state = jax.vmap(env.reset, in_axes=(0,))(video_reset_rngs)

            # Fresh hidden states and done flags (zeros, matching structure)
            video_hstate = jnp.zeros_like(hstate_v)
            video_done = jax.tree_util.tree_map(jnp.zeros_like, done_v)

            # Build video runner state (uses current policy weights)
            rng_video2, _ = jax.random.split(rng_video)
            video_runner = (train_state_v, video_env_state, video_obsv, video_done, video_hstate, rng_video2)

            # Run video rollout — result state is discarded
            _, (ego_frames, full_map_frames) = jax.lax.scan(
                _video_env_step, video_runner, None, video_length
            )

            # Restore training state with updated RNG (video state discarded)
            state = (train_state_v, env_state_v, obsv_v, done_v, hstate_v, rng_continue)

            # ego_frames: (VIDEO_LENGTH, num_agents, H, W, 3)
            # full_map_frames: (VIDEO_LENGTH, map_H, map_W, 3)

            def save_video_callback(ego_frames, full_map_frames, step):
                run_out_path = os.path.join('./', wandb.run.id, 'videos')
                os.makedirs(run_out_path, exist_ok=True)
                step_int = int(step)

                # Save per-agent ego videos
                num_agents = ego_frames.shape[1]
                wandb_videos = {}
                for agent_idx in range(num_agents):
                    agent_frames = np.asarray(ego_frames[:, agent_idx]).astype(np.uint8)
                    video_path = os.path.join(run_out_path, f'ego_agent_{agent_idx}_{step_int}.mp4')
                    try:
                        imageio.mimsave(video_path, agent_frames, fps=15, macro_block_size=1)
                        wandb_videos[f"video/ego_agent_{agent_idx}"] = wandb.Video(video_path, fps=15, format="mp4")
                        print(f'Saved ego video: {video_path}')
                    except Exception as e:
                        print(f'Failed to save ego video for agent {agent_idx}: {e}')

                # Save full map video
                full_map_np = np.asarray(full_map_frames).astype(np.uint8)
                fullmap_path = os.path.join(run_out_path, f'full_map_{step_int}.mp4')
                try:
                    imageio.mimsave(fullmap_path, full_map_np, fps=15, macro_block_size=1)
                    wandb_videos["video/full_map"] = wandb.Video(fullmap_path, fps=15, format="mp4")
                    print(f'Saved full map video: {fullmap_path}')
                except Exception as e:
                    print(f'Failed to save full map video: {e}')

                # Log all videos to wandb in one call
                if wandb_videos:
                    wandb.log(wandb_videos, step=step_int)

            jax.debug.callback(save_video_callback, ego_frames, full_map_frames, update_steps)

            runner_state = (state, update_steps)

            # Log model weights
            def save_weights_callback(weights, iter):
                weights_flat = jax.tree.flatten(weights)
                run_out_path = os.path.join('./', wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
                weight_filename = os.path.join(run_out_path, 'weights_{}.csv'.format(iter))
                weight_file = open(weight_filename, 'w')
                weights_params = weights['params']

                def save_weight_dict(curr_value, key_string=''):
                    if type(curr_value) != dict:
                        np.savetxt(weight_file, np.transpose(curr_value), delimiter=',', fmt='%f', header=key_string)
                        return True
                    else:
                        for key in curr_value.keys():
                            save_weight_dict(curr_value[key], key_string + '/' + key)
                    return True

                save_weight_dict(weights_params)

                print('Saving weights in file', weight_filename)

            # TODO make weight saving work
            #jax.debug.callback(save_weights_callback, runner_state[0].params, runner_state[-1])

            # Then, update (training)
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config["LOGGING_UPDATES_INTERVAL"]
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            init_done,  # Keep as dict for consistency with env.step output
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_plot, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "seperate-ippo-rnn")
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    env = make_craftax_env_from_name(env_name)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config["RUN_NAME"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Name of the config YAML file (in baselines/config/)")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "config", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    single_run(config)


if __name__ == "__main__":
    main()
