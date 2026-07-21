from typing import List

import jax.numpy as jnp
from craftax_coop.craftax_state import EnvState, StaticEnvParams
from craftax_coop.constants import *

def compute_score(state: EnvState, done: bool, static_params: StaticEnvParams):
    achievements = state.achievements * done * 100.0
    info = {}
    # Single-level (fixed dungeon level 2) run:
    # Skip achievements that are structurally unreachable in this setup.
    excluded_achievements_level2 = {
        # Level transitions are disabled in single-level mode.
        "ENTER_GNOMISH_MINES",
        "ENTER_DUNGEON",
        "ENTER_SEWERS",
        "ENTER_VAULT",
        "ENTER_TROLL_MINES",
        "ENTER_FIRE_REALM",
        "ENTER_ICE_REALM",
        "ENTER_GRAVEYARD",
        # Mobs/bosses from other levels.
        # On fixed level 2, FLOOR_MOB_MAPPING only spawns:
        # passive=snail, melee=orc soldier, ranged=orc mage.
        "DEFEAT_ZOMBIE",
        "DEFEAT_SKELETON",
        "DEFEAT_GNOME_WARRIOR",
        "DEFEAT_GNOME_ARCHER",
        "DEFEAT_LIZARD",
        "DEFEAT_KOBOLD",
        "DEFEAT_KNIGHT",
        "DEFEAT_ARCHER",
        "DEFEAT_TROLL",
        "DEFEAT_DEEP_THING",
        "DEFEAT_PIGMAN",
        "DEFEAT_FIRE_ELEMENTAL",
        "DEFEAT_FROST_TROLL",
        "DEFEAT_ICE_ELEMENTAL",
        "DAMAGE_NECROMANCER",
        "DEFEAT_NECROMANCER",
        # Passive mobs not present on fixed level 2.
        "EAT_COW",
        "EAT_BAT",
        # Action-disabled / unreachable progression in this setup.
        "DRINK_POTION",
        "LEARN_SPELL",
        "CAST_SPELL",
        "ENCHANT_SWORD",
        "ENCHANT_ARMOUR",
        "MAKE_IRON_ARMOUR",
        "MAKE_DIAMOND_ARMOUR",
        # Bow is only looted from the first chest on level 1.
        "FIND_BOW",
        "FIRE_BOW",
        # No source of sword level >=3 with current single-level action mask.
        "MAKE_IRON_SWORD",
        "MAKE_DIAMOND_SWORD",
        # Currently never set in game logic.
        "GIVE_ITEM",
        "Random_1",
        "Random_2",
    }
    for achievement in Achievement:
        if achievement.name in excluded_achievements_level2:
            continue
        achievement_name = f"Achievements/{achievement.name.lower()}"
        info[achievement_name] = achievements[:, achievement.value]

    # Global episode counters (broadcast scalar to match player dimension)
    info["Trade/total_trades"] = jnp.full(static_params.player_count, state.trade_count, dtype=jnp.float32)
    info["Trade/food_trades"] = jnp.full(static_params.player_count, state.food_trade_count, dtype=jnp.float32)
    info["Trade/drink_trades"] = jnp.full(static_params.player_count, state.drink_trade_count, dtype=jnp.float32)
    info["Revive/revives"] = jnp.full(static_params.player_count, state.revives, dtype=jnp.float32)

    # Per-agent episode counters
    info["Trade/trades_given"] = state.trade_give_count.astype(jnp.float32)
    info["Trade/trades_received"] = state.trade_receive_count.astype(jnp.float32)
    info["Revive/revives_given"] = state.revive_as_reviver_count.astype(jnp.float32)
    info["Revive/revives_received"] = state.revive_as_revived_count.astype(jnp.float32)

    # Team kill metrics (broadcast to match player dimension)
    for t in range(static_params.num_teams):
        info[f"Combat/team_{t}_kills"] = jnp.full(static_params.player_count, state.team_kills[t], dtype=jnp.float32)
        info[f"Combat/team_{t}_damage_dealt"] = jnp.full(static_params.player_count, state.damage_dealt_to_other_team[t], dtype=jnp.float32)

    # Per-agent metrics
    info["Reward/individual_reward"] = state.individual_reward_return.astype(jnp.float32)
    info["Movement/walking_distance"] = state.walking_distance.astype(jnp.float32)
    # Episode-average Manhattan distance from each agent to its own spawn position.
    # sum_distance_to_spawn accumulates per-step |pos - spawn| inside craftax_step;
    # dividing by timestep (steps elapsed in current episode) gives the running mean,
    # so reading this at episode end (done=True) yields the average over the episode.
    ep_steps = jnp.maximum(state.timestep.astype(jnp.float32), 1.0)
    info["Movement/distance_to_spawn"] = (state.sum_distance_to_spawn / ep_steps).astype(jnp.float32)
    info["Combat/damage_taken_melee"] = state.damage_taken_melee.astype(jnp.float32)
    info["Combat/damage_taken_health_food"] = state.damage_taken_health_food.astype(jnp.float32)
    info["Combat/damage_taken_health_drink"] = state.damage_taken_health_drink.astype(jnp.float32)
    info["Combat/damage_taken_health_energy"] = state.damage_taken_health_energy.astype(jnp.float32)
    info["Combat/damage_taken_health_other"] = state.damage_taken_health_other.astype(jnp.float32)
    info["Combat/damage_taken_ff"] = state.damage_taken_ff.astype(jnp.float32)
    return info


def compute_step_event_info(state: EnvState):
    return {
        "trade_give": state.log_trade_give.astype(jnp.float32),
        "trade_receive": state.log_trade_receive.astype(jnp.float32),
        "trade_give_material_id": state.log_trade_give_material_id.astype(jnp.float32),
        "trade_receive_material_id": state.log_trade_receive_material_id.astype(jnp.float32),
        "trade_give_partner_id": state.log_trade_give_partner_id.astype(jnp.float32),
        "trade_receive_partner_id": state.log_trade_receive_partner_id.astype(jnp.float32),
        "revive_as_reviver": state.log_revive_as_reviver.astype(jnp.float32),
        "revive_as_revived": state.log_revive_as_revived.astype(jnp.float32),
        "revive_partner_id": state.log_revive_partner_id.astype(jnp.float32),
        "melee_kills": state.log_melee_kills.astype(jnp.float32),
        "predator_hit": state.log_predator_hit.astype(jnp.float32),
        "auto_respawned": state.log_auto_respawned.astype(jnp.float32),
        "target_tile_x": (state.player_position + DIRECTIONS[state.player_direction])[:, 0].astype(jnp.float32),
        "target_tile_y": (state.player_position + DIRECTIONS[state.player_direction])[:, 1].astype(jnp.float32),
    }
