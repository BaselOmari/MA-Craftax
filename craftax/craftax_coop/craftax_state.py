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
    wood_trade_count: int
    same_trade_count: int
    revives: int
    revive_cooldown_until: jnp.ndarray  # (player_count,) earliest timestep when each agent can be revived again
    ff_damage_dealt: float
    team_kills: jnp.ndarray  # (num_teams,) array: kills against other teams, indexed by killer's team
    walking_distance: jnp.ndarray  # (player_count,) cumulative Manhattan distance
    ticks_moved: jnp.ndarray  # (player_count,) ticks where agent actually moved (step_distance > 0)
    ticks_tried_moving: jnp.ndarray  # (player_count,) ticks where agent chose a move action (incl. blocked)
    damage_taken_total: jnp.ndarray  # (player_count,) cumulative damage taken from all sources
    damage_taken_melee: jnp.ndarray  # (player_count,) cumulative mob melee damage taken
    damage_taken_ranged: jnp.ndarray  # (player_count,) cumulative ranged/projectile damage taken
    damage_taken_health: jnp.ndarray  # (player_count,) cumulative health/intrinsic/potion damage taken
    damage_taken_health_food: jnp.ndarray  # (player_count,) cumulative health damage attributed to empty food
    damage_taken_health_drink: jnp.ndarray  # (player_count,) cumulative health damage attributed to empty drink
    damage_taken_health_energy: jnp.ndarray  # (player_count,) cumulative health damage attributed to empty energy
    damage_taken_health_other: jnp.ndarray  # (player_count,) cumulative health damage from non-necessity sources (e.g., potions)
    damage_taken_ff: jnp.ndarray  # (player_count,) cumulative friendly-fire (player-vs-player) damage taken
    ticks_food_empty: jnp.ndarray  # (player_count,) cumulative steps with food == 0
    ticks_drink_empty: jnp.ndarray  # (player_count,) cumulative steps with drink == 0
    ticks_energy_empty: jnp.ndarray  # (player_count,) cumulative steps with energy == 0 (and not sleeping)
    steps_alive: jnp.ndarray  # (player_count,) cumulative steps each agent was alive
    team_alive_time: jnp.ndarray  # (num_teams,) cumulative steps where at least one team member alive
    damage_dealt_to_other_team: jnp.ndarray  # (num_teams,) cumulative damage dealt BY this team TO other teams

    # Misc Metrics
    all_necessities_frac: jnp.ndarray

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
    terminate_on_any_death: bool = False  # If True, any single agent death ends the whole episode immediately.
    reviving_cooldown_steps: int = 0  # Steps a revived agent must wait before they can be revived again.
    all_team_alive_bonus: float = 0.0  # Bonus added to shared_reward when all members of an agent's team are alive.

    # Team Spawning Parameters
    min_team_spawn_distance: int = 15

    # Trading proximity (square/Chebyshev radius in tiles).
    # before: Default keeps the old FOV-box behavior for OBS_DIM=(9,11): row<=5, col<=6,
    # approximated as a single square radius
    trade_radius: int = 10


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

    # Mobs Per Player
    max_melee_mobs: int = 2
    max_passive_mobs: int = 12
    max_growing_plants: int = 10
    max_ranged_mobs: int = 0
    max_mob_projectiles: int = 3
    max_player_projectiles: int = 3

    # Rate at which player hunger increases per tick (multiplied with base rate)
    hunger_increase_rate: float = 1.0
