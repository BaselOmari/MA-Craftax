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
from typing import Dict, Tuple
from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from craftax_ma.constants import *
from craftax_ma.craftax_state import EnvState, EnvParams, StaticEnvParams
from craftax_ma.envs.common import compute_score
from craftax_ma.game_logic import craftax_step
from craftax_ma.renderer.renderer_symbolic import render_craftax_symbolic
from craftax_ma.util.game_logic_utils import has_beaten_boss
from craftax_ma.world_gen.world_gen import generate_world


class CraftaxMASymbolicEnv(MultiAgentEnv):
    def __init__(self, num_agents: int = 2):
        self.num_agents = num_agents
        self.static_env_params = self.default_static_params()

        self.agents = [
            f"agent_{i}" for i in range(self.static_env_params.player_count)
        ]
        self.action_spaces = {name: self.action_shape() for name in self.agents}
        self.observation_spaces = {name: self.observation_shape() for name in self.agents}


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, _=None) -> Tuple[Dict[str, chex.Array], EnvState]:
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
        obs_sym = lax.stop_gradient(
            render_craftax_symbolic(
                state, 
                self.static_env_params,
            )
        )
        obs = {n:o for n,o in zip(self.agents, obs_sym)}
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
    
    def get_flat_map_obs_shape(self):
        num_mob_classes = 5
        num_mob_types = 8
        num_blocks = len(BlockType)
        num_items = len(ItemType)
        num_teammate_map = 2 # 1 bit indicates location and another indicates dead/alive
        light_map = 1

        return (
            OBS_DIM[0] *
            OBS_DIM[1] *
            (num_teammate_map + num_blocks + num_items + num_mob_classes * num_mob_types + light_map)
        )

    def get_inventory_obs_shape(self):
        num_inventory = 16
        num_potions = 6
        num_intrinsics = 9
        num_directions = 4
        num_armour = 4
        num_armour_enchantments = 4
        num_special_values = 3
        num_special_level_values = 4
        return num_inventory + num_potions + num_directions + num_intrinsics + num_armour + num_armour_enchantments + num_special_values + num_special_level_values
    
    def observation_shape(self) -> spaces.Box:
        obs_shape = (
            self.get_flat_map_obs_shape() + 
            self.get_inventory_obs_shape()
        )

        return spaces.Box(
            0.0,
            1.0,
            (obs_shape,),
            dtype=jnp.int32,
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
