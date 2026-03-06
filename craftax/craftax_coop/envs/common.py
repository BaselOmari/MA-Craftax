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

    # Trade metrics (broadcast scalar to match player dimension)
    info["Trade/total_trades"] = jnp.full(static_params.player_count, state.trade_count, dtype=jnp.float32)
    info["Trade/food_trades"] = jnp.full(static_params.player_count, state.food_trade_count, dtype=jnp.float32)
    info["Trade/drink_trades"] = jnp.full(static_params.player_count, state.drink_trade_count, dtype=jnp.float32)
    info["Trade/wood_trades"] = jnp.full(static_params.player_count, state.wood_trade_count, dtype=jnp.float32)
    info["Trade/same_subclass_trades"] = jnp.full(static_params.player_count, state.same_trade_count, dtype=jnp.float32)
    diff_trade_count = jnp.maximum(0, state.trade_count - state.same_trade_count)
    info["Trade/diff_subclass_trades"] = jnp.full(static_params.player_count, diff_trade_count, dtype=jnp.float32)

    # Team kill metrics (broadcast to match player dimension)
    for t in range(static_params.num_teams):
        info[f"Combat/team_{t}_kills"] = jnp.full(static_params.player_count, state.team_kills[t], dtype=jnp.float32)
        info[f"Combat/team_{t}_damage_dealt"] = jnp.full(static_params.player_count, state.damage_dealt_to_other_team[t], dtype=jnp.float32)

    # Per-agent metrics
    info["Movement/walking_distance"] = state.walking_distance.astype(jnp.float32)
    info["Movement/ticks_moved"] = state.ticks_moved.astype(jnp.float32)
    info["Movement/ticks_tried_moving"] = state.ticks_tried_moving.astype(jnp.float32)
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

    # Alive tracking: per-agent alive_ratio = steps_alive / team_alive_time
    agents_per_team = len(static_params.team_composition)
    agent_team_ids = jnp.arange(static_params.player_count) // agents_per_team
    agent_team_alive_time = state.team_alive_time[agent_team_ids]
    alive_ratio = jnp.where(agent_team_alive_time > 0, state.steps_alive / agent_team_alive_time, 0.0)
    info["Alive/alive_ratio"] = alive_ratio.astype(jnp.float32)
    for t in range(static_params.num_teams):
        info[f"Alive/team_{t}_alive_time"] = jnp.full(static_params.player_count, state.team_alive_time[t], dtype=jnp.float32)

    # Revives (broadcast scalar)
    info["Overview/revives"] = jnp.full(static_params.player_count, state.revives, dtype=jnp.float32)

    return info
