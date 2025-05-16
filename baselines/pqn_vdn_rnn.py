"""
Code is adapted from the PQN-VDN-RNN from the PureJaxQL repository (https://github.com/mttga/purejaxql) 
Credit goes to the original authors: Gallici et al.
"""

# ===========================
# Imports and Configuration
# ===========================
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import wandb

from jaxmarl.wrappers.baselines import (
    LogWrapper,
    CTRolloutManager,
)
from craftax.craftax_env import make_craftax_env_from_name

# ===========================
# Model Definitions
# ===========================
class ScannedRNN(nn.Module):
    @partial(
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
        hidden_size = rnn_state[0].shape[-1]

        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        rnn_state = jax.tree_util.tree_map(
            lambda init, old: jnp.where(resets[:, np.newaxis], init, old),
            init_rnn_state,
            rnn_state,
        )

        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )

class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    num_layers: int = 4
    norm_type: str = "layer_norm"
    dueling: bool = False

    @nn.compact
    def __call__(self, hidden, x, dones, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        rnn_in = (x, dones)
        hidden, x = ScannedRNN()(hidden, rnn_in)

        if self.dueling:
            adv = nn.Dense(self.action_dim)(x)
            val = nn.Dense(1)(x)
            q_vals = val + adv - jnp.mean(adv, axis=-1, keepdims=True)
        else:
            q_vals = nn.Dense(self.action_dim)(x)

        return hidden, q_vals

# ===========================
# Data Structures and Utilities
# ===========================
@chex.dataclass(frozen=True)
class Transition:
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    avail_actions: chex.Array
    q_vals: chex.Array

class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

# ===========================
# Training Function
# ===========================
def make_train(config, env):

    assert (
        config["NUM_ENVS"] % config["NUM_MINIBATCHES"] == 0
    ), "NUM_ENVS must be divisible by NUM_MINIBATCHES"

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["NUM_UPDATES_DECAY"] = (
        config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES_DECAY"],
    )

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            random_actions,
            greedy_actions,
        )
        return chosen_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        log_env = LogWrapper(env)
        wrapped_env = CTRolloutManager(
            log_env, batch_size=config["NUM_ENVS"], preprocess_obs=True
        )

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_size=config["HIDDEN_SIZE"],
            num_layers=config["NUM_LAYERS"],
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            dueling=config.get("DUELING", False),
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            network_variables = network.init(rng, init_hs, *init_x, train=False)

            lr_scheduler = optax.linear_schedule(
                config["LR"],
                1e-20,
                config["NUM_EPOCHS"]
                * config["NUM_MINIBATCHES"]
                * config["NUM_UPDATES_DECAY"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):
            train_state, memory_transitions, expl_state, rng = runner_state

            # === SAMPLE PHASE ===
            def _step_env(carry, _):
                expl_state, rng = carry
                hs, last_obs, last_dones, env_state = expl_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, None))(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _dones,
                    False,
                )
                q_vals = q_vals.squeeze(axis=1)

                avail_actions = wrapped_env.get_valid_actions(env_state)

                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                new_action = unbatchify(new_action)

                new_obs, new_env_state, reward, new_done, info = wrapped_env.batch_step(
                    rng_s, env_state, new_action
                )

                transition = Transition(
                    last_hs=hs,
                    obs=batchify(last_obs),
                    action=batchify(new_action),
                    reward=config.get("REW_SCALE", 1) * reward["__all__"][np.newaxis],
                    done=new_done["__all__"][np.newaxis],
                    last_done=batchify(last_dones),
                    avail_actions=batchify(avail_actions),
                    q_vals=q_vals,
                )
                return ((new_hs, new_obs, new_done, new_env_state), rng), (transition, info)

            rng, _rng = jax.random.split(rng)
            (expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps + config["NUM_STEPS"] * config["NUM_ENVS"]
            )

            memory_transitions = jax.tree.map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"]:], y], axis=0),
                memory_transitions,
                transitions,
            )

            # === NETWORK UPDATE PHASE ===
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):

                    train_state, rng = carry
                    hs = jax.tree_util.tree_map(lambda x: x[0].reshape(-1, config["HIDDEN_SIZE"]), minibatch.last_hs)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                    )
                    agent_in = jax.tree.map(
                        lambda x: x.reshape(x.shape[0], -1, *x.shape[3:]), agent_in
                    )

                    def _compute_targets(last_q, q_vals, reward, done):
                        def _get_target(lambda_returns_and_next_q, rew_q_done):
                            reward, q, done = rew_q_done
                            lambda_returns, next_q = lambda_returns_and_next_q
                            target_bootstrap = (
                                reward + config["GAMMA"] * (1 - done) * next_q
                            )
                            delta = lambda_returns - next_q
                            lambda_returns = (
                                target_bootstrap
                                + config["GAMMA"] * config["LAMBDA"] * delta
                            )
                            lambda_returns = (1 - done) * lambda_returns + done * reward
                            next_q = jnp.max(q, axis=-1)
                            next_q = jnp.sum(next_q, axis=0)
                            return (lambda_returns, next_q), lambda_returns

                        lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                        last_q = jnp.max(q_vals[-1], axis=-1)
                        last_q = jnp.sum(last_q, axis=0)
                        _, targets = jax.lax.scan(
                            _get_target,
                            (lambda_returns, last_q),
                            jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
                            reverse=True,
                        )
                        targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                        return targets

                    def _loss_fn(params):
                        (_, q_vals), updates = partial(
                            network.apply, train=True, mutable=["batch_stats"]
                        )(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            hs,
                            *agent_in,
                        )
                        q_vals = q_vals.reshape(
                            q_vals.shape[0], env.num_agents, -1, q_vals.shape[-1]
                        )
                        q_target = jax.lax.stop_gradient(q_vals)
                        unavailable_actions = 1 - minibatch.avail_actions
                        valid_q_vals = q_target - (unavailable_actions * 1e10)

                        last_q = valid_q_vals[-1].max(axis=-1)
                        last_q = last_q.sum(axis=0)
                        target = _compute_targets(
                            last_q,
                            valid_q_vals[:-1],
                            minibatch.reward[:-1, 0],
                            minibatch.done[:-1, 0],
                        ).reshape(-1)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)
                        vdn_chosen_action_qvals = jnp.sum(chosen_action_qvals, axis=1)[:-1].reshape(-1)

                        loss = 0.5 * jnp.mean(
                            (vdn_chosen_action_qvals - jax.lax.stop_gradient(target)) ** 2
                        )
                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = jax.random.permutation(rng, x, axis=2)
                    x = x.reshape(
                        *x.shape[:2], config["NUM_MINIBATCHES"], -1, *x.shape[3:]
                    )
                    new_order = [2, 0, 1, 3] + list(range(4, x.ndim))
                    x = jnp.transpose(x, new_order)
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree.map(
                    lambda x: preprocess_transition(x, _rng),
                    memory_transitions,
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), minibatches
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
            }

            def callback(metrics, infos):
                to_log = metrics
                if infos["returned_episode"].any():
                    to_log.update(jax.tree.map(
                        lambda x: x[infos["returned_episode"]].mean(),
                        infos["user_info"]
                    ))
                    to_log["episode_lengths"] = infos["returned_episode_lengths"][infos["returned_episode"]].mean()
                    to_log["episode_returns"] = infos["returned_episode_returns"][infos["returned_episode"]].mean()

                print(to_log)
                wandb.log(to_log, step=metrics["update_steps"])

            jax.debug.callback(callback, metrics, infos)

            runner_state = (
                train_state,
                memory_transitions,
                expl_state,
                rng,
            )

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {
            agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
            for agent in env.agents + ["__all__"]
        }
        init_hs = ScannedRNN.initialize_carry(
            config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
        )
        expl_state = (init_hs, obs, init_dones, env_state)

        # Fill memory window
        def _random_step(carry, _):
            expl_state, rng = carry
            hs, last_obs, last_dones, env_state = expl_state
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = batchify(last_obs)[:, np.newaxis]
            _dones = batchify(last_dones)[:, np.newaxis]
            avail_actions = wrapped_env.get_valid_actions(env_state)
            new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, None))(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                hs,
                _obs,
                _dones,
                False,
            )
            _rngs = jax.random.split(rng_a, env.num_agents)
            new_action = {
                agent: wrapped_env.batch_sample(_rngs[i], agent)
                for i, agent in enumerate(env.agents)
            }
            new_obs, new_env_state, reward, new_done, info = wrapped_env.batch_step(
                rng_s, env_state, new_action
            )
            transition = Transition(
                last_hs=hs,
                obs=batchify(last_obs),
                action=batchify(new_action),
                reward=reward["__all__"][np.newaxis],
                done=new_done["__all__"][np.newaxis],
                last_done=batchify(last_dones),
                avail_actions=batchify(avail_actions),
                q_vals=q_vals.squeeze(axis=1),
            )
            return ((new_hs, new_obs, new_done, new_env_state), rng), transition

        rng, _rng = jax.random.split(rng)
        (expl_state, rng), memory_transitions = jax.lax.scan(
            _random_step,
            (expl_state, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )

        runner_state = (train_state, memory_transitions, expl_state, _rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "pqn-vdn-rnn")
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
    # Optional: config from CLI or YAML can be loaded here for flexibility
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="YAML config file path", default=None)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "config", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    single_run(config)

if __name__ == "__main__":
    main()
