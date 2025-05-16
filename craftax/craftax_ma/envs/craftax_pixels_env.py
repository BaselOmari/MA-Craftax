# ===========================
# Imports and Configuration
# ===========================
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chex
import jax
from functools import partial
from jax import lax
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from typing import Dict, Tuple

from craftax_ma.constants import *
from craftax_ma.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax_ma.envs.common import compute_score
from craftax_ma.game_logic import craftax_step
from craftax_ma.renderer.renderer_pixels import render_craftax_pixels
from craftax_ma.util.game_logic_utils import has_beaten_boss
from craftax_ma.world_gen.world_gen import generate_world

class CraftaxMAPixelsEnv(MultiAgentEnv):
    def __init__(self, num_agents: int = 2):  # Changed default to 2 for consistency
        self.num_agents = num_agents
        self.static_env_params = self.default_static_params()
        self.pixel_size = BLOCK_PIXEL_SIZE_AGENT

        self.agents = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]
        self.action_spaces = {name: self.action_shape() for name in self.agents}
        self.observation_spaces = {name: self.observation_shape() for name in self.agents}

        self.player_specific_textures = load_player_specific_textures(
            TEXTURES[self.pixel_size],
            self.static_env_params.player_count
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> Tuple[Dict[str, chex.Array], EnvState]:  # Added _=None for parity
        state = generate_world(key, self.default_params, self.static_env_params)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], EnvState, Dict[str, float], Dict[str, bool], Dict]:
        actions = jnp.array(list(actions.values()))
        state, reward = craftax_step(key, state, actions, self.default_params, self.static_env_params)

        obs = self.get_obs(state)
        done = self.is_terminal(state, self.default_params)
        info = {}
        info["user_info"] = compute_score(state, done, self.static_env_params)
        info["discount"] = self.discount(state, self.default_params) 

        agent_rewards = {n: r for n,r in zip(self.agents, reward)}

        agent_done = {n: done for n in self.agents}
        agent_done["__all__"] = done

        return (
            obs,
            lax.stop_gradient(state),
            agent_rewards,
            agent_done,
            info,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: EnvState) -> Dict[str, chex.Array]:
        pixels = lax.stop_gradient(
            render_craftax_pixels(
                state, 
                self.pixel_size, 
                self.static_env_params,
                self.player_specific_textures
            ) / 255.0
        )
        obs = {n: o for n, o in zip(self.agents, pixels)}
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: EnvState) -> Dict[str, chex.Array]:
        aa = jnp.full(len(Action), True)
        return {
            agent: aa[i]
            for i, agent in enumerate(self.agents)
        }

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    @staticmethod
    def default_static_params() -> StaticEnvParams:
        return StaticEnvParams()
    
    def action_shape(self) -> spaces.Discrete:
        return spaces.Discrete(len(Action))

    def observation_shape(self) -> spaces.Box:
        map_height = OBS_DIM[0]
        inventory_height = INVENTORY_OBS_HEIGHT
        teammate_dashboard_height = (self.static_env_params.player_count+1)//2
        return spaces.Box(
            0.0,
            1.0,
            (
                OBS_DIM[1] * self.pixel_size,
                (map_height + inventory_height + teammate_dashboard_height) * self.pixel_size,
                3,
            ),
            dtype=jnp.float32,
        )
    
    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        done_steps = state.timestep >= params.max_timesteps
        is_dead = jnp.logical_not(state.player_alive).all()
        defeated_boss = has_beaten_boss(state, self.static_env_params)
        is_terminal = jnp.logical_or(is_dead, done_steps)
        is_terminal = jnp.logical_or(is_terminal, defeated_boss)
        return is_terminal
    
    def discount(self, state, params) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)
