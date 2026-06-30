from dataclasses import dataclass
from typing import Tuple, Any

import jax
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Inventory:
    wood: jnp.ndarray
    stone: jnp.ndarray
    coal: jnp.ndarray
    iron: jnp.ndarray
    diamond: jnp.ndarray
    sapling: jnp.ndarray
    pickaxe: jnp.ndarray
    sword: jnp.ndarray
    bow: jnp.ndarray
    arrows: jnp.ndarray
    armour: jnp.ndarray
    torches: jnp.ndarray
    ruby: jnp.ndarray
    sapphire: jnp.ndarray
    potions: jnp.ndarray
    books: jnp.ndarray


@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: jnp.ndarray
    mask: jnp.ndarray
    attack_cooldown: jnp.ndarray
    type_id: jnp.ndarray


# @struct.dataclass
# class Projectiles(Mobs):
#     directions: jnp.ndarray
#     lifetimes: jnp.ndarray


@struct.dataclass
class EnvState:
    map: jnp.ndarray
    item_map: jnp.ndarray
    mob_map: jnp.ndarray
    light_map: jnp.ndarray
    down_ladders: jnp.ndarray
    up_ladders: jnp.ndarray
    chests_opened: jnp.ndarray
    monsters_killed: jnp.ndarray

    player_position: jnp.ndarray
    player_spawn_position: jnp.ndarray
    player_level: int
    player_direction: jnp.ndarray
    player_alive: jnp.ndarray

    # Intrinsics
    player_health: jnp.ndarray
    player_food: jnp.ndarray
    player_drink: jnp.ndarray
    player_energy: jnp.ndarray
    player_mana: jnp.ndarray
    is_sleeping: jnp.ndarray
    is_resting: jnp.ndarray

    # Second order intrinsics
    player_recover: jnp.ndarray
    player_hunger: jnp.ndarray
    player_thirst: jnp.ndarray
    player_fatigue: jnp.ndarray
    player_recover_mana: jnp.ndarray 

    # Attributes
    player_xp: jnp.ndarray
    player_dexterity: jnp.ndarray
    player_strength: jnp.ndarray
    player_intelligence: jnp.ndarray
    player_specialization: jnp.ndarray
    player_sc: jnp.ndarray # subclasses

    # Request Info
    request_duration: jnp.ndarray
    request_type: jnp.ndarray

    inventory: Inventory

    melee_mobs: Mobs
    passive_mobs: Mobs
    ranged_mobs: Mobs

    mob_projectiles: Mobs
    mob_projectile_directions: jnp.ndarray
    mob_projectile_owners: jnp.ndarray
    player_projectiles: Mobs
    player_projectile_directions: jnp.ndarray
    player_projectile_owners: jnp.ndarray

    growing_plants_positions: jnp.ndarray
    growing_plants_age: jnp.ndarray
    growing_plants_mask: jnp.ndarray

    potion_mapping: jnp.ndarray
    learned_spells: jnp.ndarray

    sword_enchantment: jnp.ndarray
    bow_enchantment: jnp.ndarray
    armour_enchantments: jnp.ndarray

    boss_progress: int
    boss_timesteps_to_spawn_this_round: int

    light_level: float

    achievements: jnp.ndarray

    state_rng: Any

    timestep: int

    # cooperation metrics
    trade_count: int
    food_trade_count: int
    drink_trade_count: int
    revives: int
    revive_cooldown_until: jnp.ndarray  # (player_count,) earliest timestep when each agent can be revived again
    trade_give_count: jnp.ndarray  # (player_count,) cumulative successful trades given by each agent
    trade_receive_count: jnp.ndarray  # (player_count,) cumulative successful trades received by each agent
    revive_as_reviver_count: jnp.ndarray  # (player_count,) cumulative revives performed by each agent
    revive_as_revived_count: jnp.ndarray  # (player_count,) cumulative times each agent was revived
    team_kills: jnp.ndarray  # (num_teams,) array: kills against other teams, indexed by killer's team
    walking_distance: jnp.ndarray  # (player_count,) cumulative Manhattan distance
    sum_distance_to_spawn: jnp.ndarray  # (player_count,) sum of per-step Manhattan distance from spawn; divide by timestep for episode average
    damage_taken_melee: jnp.ndarray  # (player_count,) cumulative mob melee damage taken
    damage_taken_health_food: jnp.ndarray  # (player_count,) cumulative health damage attributed to empty food
    damage_taken_health_drink: jnp.ndarray  # (player_count,) cumulative health damage attributed to empty drink
    damage_taken_health_energy: jnp.ndarray  # (player_count,) cumulative health damage attributed to empty energy
    damage_taken_health_other: jnp.ndarray  # (player_count,) cumulative health damage from non-necessity sources (e.g., potions)
    damage_taken_ff: jnp.ndarray  # (player_count,) cumulative friendly-fire (player-vs-player) damage taken
    consecutive_dead_steps: jnp.ndarray  # (player_count,) consecutive timesteps each agent has remained dead
    damage_dealt_to_other_team: jnp.ndarray  # (num_teams,) cumulative damage dealt BY this team TO other teams
    individual_reward_return: jnp.ndarray  # (player_count,) cumulative per-agent individual reward path
    log_trade_give: jnp.ndarray  # (player_count,) 1 if agent successfully gave a trade item this step, else 0
    log_trade_receive: jnp.ndarray  # (player_count,) 1 if agent successfully received a trade item this step, else 0
    log_trade_give_material_id: jnp.ndarray  # (player_count,) compact material code, -1 if none
    log_trade_receive_material_id: jnp.ndarray  # (player_count,) compact material code, -1 if none
    log_trade_give_partner_id: jnp.ndarray  # (player_count,) trade recipient agent id, -1 if none
    log_trade_receive_partner_id: jnp.ndarray  # (player_count,) trade sender agent id, -1 if none
    log_revive_as_reviver: jnp.ndarray  # (player_count,) 1 if agent revived another this step, else 0
    log_revive_as_revived: jnp.ndarray  # (player_count,) 1 if agent was revived this step, else 0
    log_revive_partner_id: jnp.ndarray  # (player_count,) counterpart agent id for revive event, -1 if none
    log_melee_kills: jnp.ndarray  # (player_count,) melee mob kills credited to each agent during the current step
    log_predator_hit: jnp.ndarray  # (player_count,) 1 if agent was hit by a predator this step, else 0
    log_auto_respawned: jnp.ndarray  # (player_count,) 1 if agent auto-respawned this step, else 0

    # Per-player bounds of the player's assigned starter room, used by enable_auto_respawning.
    player_spawn_room_min: jnp.ndarray  # (player_count, 2) top-left (row, col) of the player's own start room
    player_spawn_room_max: jnp.ndarray  # (player_count, 2) bottom-right (row, col) inclusive of the player's own start room

    # Episode length cap (dynamically set by training loop).
    # Keep this float32 so reset/step states stay dtype-consistent under JAX auto-reset.
    effective_max_timesteps: float = 100000.0

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


@struct.dataclass
class EnvParams:
    max_timesteps: int = 100000
    day_length: int = 300

    melee_mob_health: int = 5
    passive_mob_health: int = 3
    ranged_mob_health: int = 3

    mob_despawn_distance: int = 500
    max_attribute: int = 5


    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)

    # Game Mode Parameters
    god_mode: bool = False
    shared_reward: bool = True
    team_based_sharing: bool = True  # If True, share rewards only within teams; if False, share across all agents
    reward_func: str = 'foraging'  # 'vanilla' or 'foraging'
    friendly_fire: bool = True
    allow_neg_reward_if_dead: bool = False  # If True, dead agents get max negative foraging step reward.
    disable_revive: bool = False  # If True, players cannot revive downed teammates -> can also be set in yaml
    terminate_on_any_death: bool = False  # If True, the episode ends once any agent has remained dead longer than terminate_on_any_death_offset.
    terminate_on_any_death_offset: int = 200  # Dead-step threshold used when terminate_on_any_death is enabled.
    reviving_cooldown_steps: int = 0  # Steps a revived agent must wait before they can be revived again.
    teammate_alive_bonus: float = 0.0  # Shared bonus per additional alive team member beyond the first alive member.
    all_team_alive_bonus: float = 0.0  # Bonus added to shared_reward when all members of an agent's team are alive.
    dead_self_penalty_weight: float = 0.0  # Per-agent penalty applied only to dead agents after shared reward aggregation.
    one_time_death_penalty_shared: float = 0.0  # One-time penalty per death, distributed to the whole team via shared reward.
    one_time_death_penalty_individual: float = 0.0  # One-time penalty per death, applied only to the dead agent (in both reward modes).
    warrior_melee_kill_reward: float = 0.0  # Bonus given to warriors per credited melee-mob kill.
    forager_melee_kill_reward: float = 0.0  # Bonus given to foragers per credited melee-mob kill.
    warrior_passive_food_gain: int = 1  # Food a warrior gains when eating a killed passive mob.
    forager_passive_food_gain: int = 3  # Food a forager gains when eating a killed passive mob.
    forager_food_capacity: int = 27  # Maximum food capacity for foragers.
    forager_predator_damage_multiplier: float = 1.0  # Multiplier for forager damage against melee/ranged mobs.
    forager_to_warrior_food_trade_reward: float = 0.0  # Bonus given to foragers for successfully feeding a warrior.
    forager_to_warrior_drink_trade_reward: float = 0.0  # Bonus given to foragers for successfully hydrating a warrior.
    trade_reward_requires_both_outside_starter_room: bool = False  # If True, food/drink trade bonuses only pay when both traders are outside their own starter rooms.
    enable_auto_respawning: bool = False  # If True, auto-revive dead agents after auto_respawn_steps.
    auto_respawn_steps: int = 50  # Dead-step threshold for auto-respawn when enabled.
    restrict_auto_respawning_to_spawn_room: bool = True  # If True, auto-respawn only triggers when the dead agent is inside its own starter room.

    # Team Spawning Parameters
    min_team_spawn_distance: int = 15
    initial_predators_spawn_in_warrior_rooms_only: bool = False  # If True, t=0 melee predators are only seeded in the assigned warrior start room.

    # Trading proximity in tiles.
    trade_radius: int = 18
    trade_radius_shape: str = "square"  # "square" (Chebyshev) or "circle" (Euclidean)


@struct.dataclass
class StaticEnvParams:
    map_size: Tuple[int, int] = (96, 96)
    num_levels: int = 9
    player_count: int = 6

    # Team Configuration
    # team_composition: tuple of Specialization values defining roles per team
    # e.g. (1, 1, 2) = (FORAGER, FORAGER, WARRIOR) -> 3 agents per team
    team_composition: tuple = (1, 1, 2)
    num_teams: int = 2

    # Global mob / projectile / plant caps (no longer scaled by player_count)
    max_melee_mobs: int = 40
    max_passive_mobs: int = 105
    max_growing_plants: int = 60
    max_ranged_mobs: int = 0
    max_mob_projectiles: int = 18
    max_player_projectiles: int = 18

    corridor_width: int = 2

    # Rate at which player hunger increases per tick (multiplied with base rate)
    hunger_increase_rate: float = 1.0
