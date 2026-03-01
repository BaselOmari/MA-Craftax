from typing import List

import jax.numpy as jnp
from craftax_coop.craftax_state import EnvState, StaticEnvParams
from craftax_coop.constants import *

def compute_score(state: EnvState, done: bool, static_params: StaticEnvParams):
    achievements = state.achievements * done * 100.0
    info = {}
    for achievement in Achievement:
        achievement_name = f"Achievements/{achievement.name.lower()}"
        info[achievement_name] = achievements[:, achievement.value]

    # Log interactions between agents
    # interaction_info = log_interactions(state, done, static_params)
    # info.update(interaction_info)

    # Add trade metrics (broadcast scalar to match player dimension)
    info["Trade/total_trades"] = jnp.full(static_params.player_count, state.trade_count, dtype=jnp.float32)
    info["Trade/food_trades"] = jnp.full(static_params.player_count, state.food_trade_count, dtype=jnp.float32)
    info["Trade/drink_trades"] = jnp.full(static_params.player_count, state.drink_trade_count, dtype=jnp.float32)
    info["Trade/wood_trades"] = jnp.full(static_params.player_count, state.wood_trade_count, dtype=jnp.float32)
    info["Trade/same_subclass_trades"] = jnp.full(static_params.player_count, state.same_trade_count, dtype=jnp.float32)
    info["Trade/diff_subclass_trades"] = jnp.full(static_params.player_count, state.diff_trade_count, dtype=jnp.float32)
    
    # Add team kill metrics (broadcast to match player dimension)
    info["Combat/team_a_kills"] = jnp.full(static_params.player_count, state.team_kills[0], dtype=jnp.float32)
    info["Combat/team_b_kills"] = jnp.full(static_params.player_count, state.team_kills[1], dtype=jnp.float32)
    info["Movement/walking_distance"] = state.walking_distance.astype(jnp.float32)
    info["Combat/damage_taken_total"] = state.damage_taken_total.astype(jnp.float32)
    info["Combat/damage_taken_melee"] = state.damage_taken_melee.astype(jnp.float32)
    info["Combat/damage_taken_ranged"] = state.damage_taken_ranged.astype(jnp.float32)
    info["Combat/damage_taken_health"] = state.damage_taken_health.astype(jnp.float32)
    info["Combat/damage_taken_health_food"] = state.damage_taken_health_food.astype(jnp.float32)
    info["Combat/damage_taken_health_drink"] = state.damage_taken_health_drink.astype(jnp.float32)
    info["Combat/damage_taken_health_energy"] = state.damage_taken_health_energy.astype(jnp.float32)
    info["Combat/damage_taken_health_other"] = state.damage_taken_health_other.astype(jnp.float32)
    info["Combat/damage_taken_ff"] = state.damage_taken_ff.astype(jnp.float32)
    info["Necessities/ticks_food_empty"] = state.ticks_food_empty.astype(jnp.float32)
    info["Necessities/ticks_drink_empty"] = state.ticks_drink_empty.astype(jnp.float32)
    info["Necessities/ticks_energy_empty"] = state.ticks_energy_empty.astype(jnp.float32)
    
    return info


# def log_interactions(state: EnvState, done: bool, static_params: StaticEnvParams):
#     """
#     Log aggregated interactions between agents.
#     Sums interactions received by each agent across all interaction types.
#     Returns per-agent interaction counts to match the shape of other metrics.
#     """
#     interactions = state.interactions * done * 1.0
#     info = {}
    
#     # Aggregate interactions per agent: sum over actor and interaction type
#     # Result shape: (player_count,) - total interactions received by each agent
#     for interaction in Interaction:
#         interaction_name = interaction.name.lower()
#         # Sum over all actors for each receiver
#         per_agent = jnp.sum(interactions[:, :, interaction.value], axis=0)
#         key = f"Interactions/{interaction_name}"
#         info[key] = per_agent
    
#     return info
