import os
import pathlib
from enum import Enum
import jax.numpy as jnp
import imageio.v3 as iio
import numpy as np
from PIL import Image
from craftax_coop.util.maths_utils import get_distance_map
from environment_base.util import load_compressed_pickle, save_compressed_pickle
from flax import struct
from seaborn import husl_palette

# GAME CONSTANTS
OBS_DIM = (9, 11)
assert OBS_DIM[0] % 2 == 1 and OBS_DIM[1] % 2 == 1
MAX_OBS_DIM = max(OBS_DIM)
BLOCK_PIXEL_SIZE_HUMAN = 64
BLOCK_PIXEL_SIZE_IMG = 16
BLOCK_PIXEL_SIZE_AGENT = 10
INVENTORY_OBS_HEIGHT = 4
TEXTURE_CACHE_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(), "assets", "texture_cache.pbz2"
)

REQUEST_MAX_DURATION = 10

# DUNGEON ROOM CONSTANTS
NUM_ROOMS = 8
MIN_ROOM_SIZE = 5
MAX_ROOM_SIZE = 10


# ENUMS
class BlockType(Enum):
    INVALID = 0
    OUT_OF_BOUNDS = 1
    GRASS = 2
    WATER = 3
    STONE = 4
    TREE = 5
    WOOD = 6
    PATH = 7
    COAL = 8
    IRON = 9
    DIAMOND = 10
    CRAFTING_TABLE = 11
    FURNACE = 12
    SAND = 13
    LAVA = 14
    PLANT = 15
    RIPE_PLANT = 16
    WALL = 17
    DARKNESS = 18
    WALL_MOSS = 19
    STALAGMITE = 20
    SAPPHIRE = 21
    RUBY = 22
    CHEST = 23
    FOUNTAIN = 24
    FIRE_GRASS = 25
    ICE_GRASS = 26
    GRAVEL = 27
    FIRE_TREE = 28
    ICE_SHRUB = 29
    ENCHANTMENT_TABLE_FIRE = 30
    ENCHANTMENT_TABLE_ICE = 31
    NECROMANCER = 32
    GRAVE = 33
    GRAVE2 = 34
    GRAVE3 = 35
    NECROMANCER_VULNERABLE = 36


class ItemType(Enum):
    NONE = 0
    TORCH = 1
    LADDER_DOWN = 2
    LADDER_UP = 3
    LADDER_DOWN_BLOCKED = 4


class Action(Enum):
    NOOP = 0  #
    LEFT = 1  # a
    RIGHT = 2  # d
    UP = 3  # w
    DOWN = 4  # s
    DO = 5  # space
    SLEEP = 6  # tab
    PLACE_STONE = 7  # r
    PLACE_TABLE = 8  # t
    PLACE_FURNACE = 9  # f
    PLACE_PLANT = 10  # p
    MAKE_WOOD_PICKAXE = 11  # 1
    MAKE_STONE_PICKAXE = 12  # 2
    MAKE_IRON_PICKAXE = 13  # 3
    MAKE_WOOD_SWORD = 14  # 5
    MAKE_STONE_SWORD = 15  # 6
    MAKE_IRON_SWORD = 16  # 7
    REST = 17  # e
    DESCEND = 18  # >
    ASCEND = 19  # <
    MAKE_DIAMOND_PICKAXE = 20  # 4
    MAKE_DIAMOND_SWORD = 21  # 8
    MAKE_IRON_ARMOUR = 22  # y
    MAKE_DIAMOND_ARMOUR = 23  # u
    SHOOT_ARROW = 24  # i
    MAKE_ARROW = 25  # o
    CAST_SPELL = 26  # g
    PLACE_TORCH = 28  # j
    DRINK_POTION_RED = 29  # z
    DRINK_POTION_GREEN = 30  # x
    DRINK_POTION_BLUE = 31  # c
    DRINK_POTION_PINK = 32  # v
    DRINK_POTION_CYAN = 33  # b
    DRINK_POTION_YELLOW = 34  # n
    READ_BOOK = 35  # m
    ENCHANT_SWORD = 36  # k
    ENCHANT_ARMOUR = 37  # l
    MAKE_TORCH = 38  # [
    LEVEL_UP_DEXTERITY = 39  # ]
    LEVEL_UP_STRENGTH = 40  # -
    LEVEL_UP_INTELLIGENCE = 41  # =
    ENCHANT_BOW = 42  # ;
    REQUEST_FOOD = 43  # Backspace
    REQUEST_DRINK = 44  # Back slash
    REQUEST_WOOD = 45  # Return
    REQUEST_STONE = 46  # Right Shift
    REQUEST_IRON = 47  # Up Arrow
    REQUEST_COAL = 48  # Down Arrow
    REQUEST_DIAMOND = 49  # Left Arrow
    REQUEST_RUBY = 50  # Left Arrow
    REQUEST_SAPPHIRE = 51  # Left Arrow
    GIVE = 52  # Right Arrow
    # Player can give to all other players. (Action - GIVE) represents which player to give to.

def avail_actions_fn(num_agents):
    base_actions = [
        1,  # 0: NOOP ✅
        1,  # 1: LEFT ✅
        1,  # 2: RIGHT ✅
        1,  # 3: UP ✅
        1,  # 4: DOWN ✅
        1,  # 5: DO ✅
        1,  # 6: SLEEP ✅
        1,  # 7: PLACE_STONE ✅
        1,  # 8: PLACE_TABLE ✅
        1,  # 9: PLACE_FURNACE ✅
        1,  # 10: PLACE_PLANT ✅
        1,  # 11: MAKE_WOOD_PICKAXE ✅
        1,  # 12: MAKE_STONE_PICKAXE ✅
        0,  # 13: MAKE_IRON_PICKAXE ❌
        1,  # 14: MAKE_WOOD_SWORD ✅
        1,  # 15: MAKE_STONE_SWORD ✅
        0,  # 16: MAKE_IRON_SWORD ❌
        1,  # 17: REST ✅
        1,  # 18: DESCEND ✅
        1,  # 19: ASCEND ✅
        0,  # 20: MAKE_DIAMOND_PICKAXE ❌
        0,  # 21: MAKE_DIAMOND_SWORD ❌
        0,  # 22: MAKE_IRON_ARMOUR ❌
        0,  # 23: MAKE_DIAMOND_ARMOUR ❌
        1,  # 24: SHOOT_ARROW ✅
        1,  # 25: MAKE_ARROW ✅
        0,  # 26: CAST_SPELL ❌
        # 27 is missing from enum
        1,  # 28: PLACE_TORCH ✅
        0,  # 29: DRINK_POTION_RED ❌
        0,  # 30: DRINK_POTION_GREEN ❌
        0,  # 31: DRINK_POTION_BLUE ❌
        0,  # 32: DRINK_POTION_PINK ❌
        0,  # 33: DRINK_POTION_CYAN ❌
        0,  # 34: DRINK_POTION_YELLOW ❌
        0,  # 35: READ_BOOK ❌
        0,  # 36: ENCHANT_SWORD ❌
        0,  # 37: ENCHANT_ARMOUR ❌
        1,  # 38: MAKE_TORCH ✅
        0,  # 39: LEVEL_UP_DEXTERITY ❌
        0,  # 40: LEVEL_UP_STRENGTH ❌
        0,  # 41: LEVEL_UP_INTELLIGENCE ❌
        0,  # 42: ENCHANT_BOW ❌
        0,  # 43: REQUEST_FOOD ❌
        0,  # 44: REQUEST_DRINK ❌
        0,  # 45: REQUEST_WOOD ❌
        0,  # 46: REQUEST_STONE ❌
        0,  # 47: REQUEST_IRON ❌
        0,  # 48: REQUEST_COAL ❌
        0,  # 49: REQUEST_DIAMOND ❌
        0,  # 50: REQUEST_RUBY ❌
        0,  # 51: REQUEST_SAPPHIRE ❌
        0,  # 52: GIVE ❌
    ]

    # Add GIVE_TO_PLAYER_X for other agents (all disabled)
    extra_give_actions = [0] * (num_agents - 1)

    return jnp.array(base_actions + extra_give_actions, dtype=int)

class MobType(Enum):
    PASSIVE = 0
    MELEE = 1
    RANGED = 2
    PROJECTILE = 3


class ProjectileType(Enum):
    ARROW = 0
    DAGGER = 1
    FIREBALL = 2
    ICEBALL = 3
    ARROW2 = 4
    SLIMEBALL = 5
    FIREBALL2 = 6
    ICEBALL2 = 7


class Specialization(Enum):
    UNASSIGNED = 0
    FORAGER = 1
    WARRIOR = 2
    MINER = 3


# FLOOR MECHANICS

FLOOR_MOB_MAPPING = jnp.array(
    [
        # (passive, melee, ranged)
        jnp.array([0, 0, 0]),  # Floor 0 (overworld)
        jnp.array([2, 2, 2]),  # Floor 1 (dungeon)
        jnp.array([1, 1, 1]),  # Floor 2 (gnomish mines)
        jnp.array([2, 3, 3]),  # Floor 3 (sewers)
        jnp.array([2, 4, 4]),  # Floor 4 (vaults)
        jnp.array([1, 5, 5]),  # Floor 5 (troll mines)
        jnp.array([1, 6, 6]),  # Floor 6 (fire)
        jnp.array([1, 7, 7]),  # Floor 7 (ice)
        jnp.array([0, 0, 0]),  # Floor 8 (boss)
    ],
    dtype=jnp.int32,
)


FLOOR_MOB_SPAWN_CHANCE = jnp.array(
    [
        # (passive, melee, ranged, melee-night)
        jnp.array([0.1, 0.02, 0.05, 0.1]),  # Floor 0 (overworld)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 1 (gnomish mines)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 2 (dungeon)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 3 (sewers)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 4 (vaults)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 5 (troll mines)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 6 (fire)
        jnp.array([0.0, 0.06, 0.05, 0.0]),  # Floor 7 (ice)
        jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

# Path blocks, water, lava  (everything collides with solid blocks)
COLLISION_LAND_CREATURE = [False, True, True]
COLLISION_FLYING = [False, False, False]
COLLISION_AQUATIC = [True, False, True]
COLLISION_AMPHIBIAN = [False, False, True]


MOB_TYPE_COLLISION_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, projectile)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 0 (overworld)
        jnp.array(
            [
                COLLISION_FLYING,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 1 (gnomish mines)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 2 (dungeon)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_AMPHIBIAN,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 3 (sewers)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 4 (vaults)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_AQUATIC,
                COLLISION_FLYING,
            ]
        ),  # Floor 5 (troll mines)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
                COLLISION_FLYING,
            ]
        ),  # Floor 6 (fire)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
                COLLISION_FLYING,
            ]
        ),  # Floor 7 (ice)
        jnp.array(
            [
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_LAND_CREATURE,
                COLLISION_FLYING,
            ]
        ),  # Floor 8 (boss)
    ],
    dtype=jnp.int32,
)

NO_DAMAGE = jnp.array([0, 0, 0])
MOB_TYPE_DAMAGE_MAPPING = jnp.array(
    [
        # (-, melee, -, projectile)
        [NO_DAMAGE, [2, 0, 0], NO_DAMAGE, [2, 0, 0]],  # zombie, arrow
        [NO_DAMAGE, [4, 0, 0], NO_DAMAGE, [4, 0, 0]],  # gnome, dagger
        [NO_DAMAGE, [3, 0, 0], NO_DAMAGE, [0, 3, 0]],  # orc, fireball
        [NO_DAMAGE, [5, 0, 0], NO_DAMAGE, [0, 0, 3]],  # lizard, iceball
        [NO_DAMAGE, [6, 0, 0], NO_DAMAGE, [5, 0, 0]],  # knight, arrow2
        [NO_DAMAGE, [6, 1, 1], NO_DAMAGE, [4, 3, 3]],  # troll, slimeball
        [NO_DAMAGE, [3, 5, 0], NO_DAMAGE, [3, 5, 0]],  # pigman, fireball2
        [NO_DAMAGE, [4, 0, 5], NO_DAMAGE, [4, 0, 5]],  # ice troll, iceball2
    ],
    dtype=jnp.float32,
)

MOB_TYPE_HEALTH_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, -)
        jnp.array([3, 5, 3, 0]),  # Floor 0 (overworld)
        jnp.array([4, 7, 5, 0]),  # Floor 1 (gnomish mines)
        jnp.array([6, 9, 6, 0]),  # Floor 2 (dungeon)
        jnp.array([8, 11, 8, 0]),  # Floor 3 (sewers)
        jnp.array([0, 12, 12, 0]),  # Floor 4 (vaults)
        jnp.array([0, 20, 4, 0]),  # Floor 5 (troll mines)
        jnp.array([0, 20, 14, 0]),  # Floor 6 (fire)
        jnp.array([0, 24, 16, 0]),  # Floor 7 (ice)
        jnp.array([0, 0, 0, 0]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

NO_DEFENSE = [0, 0, 0]
MOB_TYPE_DEFENSE_MAPPING = jnp.array(
    [
        # (passive, melee, ranged, -)
        jnp.array(
            [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]
        ),  # Floor 0 (overworld)
        jnp.array(
            [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]
        ),  # Floor 1 (gnomish mines)
        jnp.array(
            [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]
        ),  # Floor 2 (dungeon)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 3 (sewers)
        jnp.array(
            [NO_DEFENSE, [0.5, 0, 0], [0.5, 0, 0], NO_DEFENSE]
        ),  # Floor 4 (vaults)
        jnp.array(
            [NO_DEFENSE, [0.2, 0, 0], [0.0, 0.0, 0.0], NO_DEFENSE]
        ),  # Floor 5 (troll mines)
        jnp.array(
            [NO_DEFENSE, [0.9, 1.0, 0.0], [0.9, 1.0, 0.0], NO_DEFENSE]
        ),  # Floor 6 (fire)
        jnp.array(
            [NO_DEFENSE, [0.9, 0.0, 1.0], [0.9, 0.0, 1.0], NO_DEFENSE]
        ),  # Floor 7 (ice)
        jnp.array([NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE]),  # Floor 8 (boss)
    ],
    dtype=jnp.float32,
)

RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_MAPPING = jnp.array(
    [
        0,  # Skeleton --> Arrow
        0,  # Gnome archer --> Arrow
        2,  # Orc mage --> Fireball
        1,  # Kobold --> Dagger
        4,  # Knight archer --> Arrow2
        5,  # Deep thing --> Slime ball
        6,  # Fire elemental --> Fireball2
        7,  # Ice elemental --> Iceball2
    ]
)


# GAME MECHANICS
MONSTERS_KILLED_TO_CLEAR_LEVEL = 8
BOSS_FIGHT_EXTRA_DAMAGE = 0.5
BOSS_FIGHT_SPAWN_TURNS = 7

DIRECTIONS = jnp.concatenate(
    (
        jnp.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32),
        jnp.zeros((11, 2), dtype=jnp.int32),
    ),
    axis=0,
)

CLOSE_BLOCKS = jnp.array(
    [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0],
        [-1, -1],
        [-1, 1],
        [1, -1],
        [1, 1],
    ],
    dtype=jnp.int32,
)

# Can't walk through these
SOLID_BLOCKS = [
    BlockType.STONE.value,
    BlockType.TREE.value,
    BlockType.COAL.value,
    BlockType.IRON.value,
    BlockType.DIAMOND.value,
    BlockType.CRAFTING_TABLE.value,
    BlockType.FURNACE.value,
    BlockType.PLANT.value,
    BlockType.RIPE_PLANT.value,
    BlockType.WALL.value,
    BlockType.WALL_MOSS.value,
    BlockType.STALAGMITE.value,
    BlockType.RUBY.value,
    BlockType.SAPPHIRE.value,
    BlockType.CHEST.value,
    BlockType.FOUNTAIN.value,
    BlockType.FIRE_TREE.value,
    BlockType.ENCHANTMENT_TABLE_FIRE.value,
    BlockType.ENCHANTMENT_TABLE_ICE.value,
    BlockType.GRAVE.value,
    BlockType.GRAVE2.value,
    BlockType.GRAVE3.value,
    BlockType.NECROMANCER.value,
]

SOLID_BLOCK_MAPPING = jnp.array(
    [(block.value in SOLID_BLOCKS) for block in BlockType], dtype=bool
)

CAN_PLACE_ITEM_BLOCKS = [
    BlockType.GRASS.value,
    BlockType.SAND.value,
    BlockType.PATH.value,
    BlockType.FIRE_GRASS.value,
    BlockType.ICE_GRASS.value,
]

CAN_PLACE_ITEM_MAPPING = jnp.array(
    [(block.value in CAN_PLACE_ITEM_BLOCKS) for block in BlockType], dtype=bool
)


# ACHIEVEMENTS
class Achievement(Enum):
    COLLECT_WOOD = 0
    PLACE_TABLE = 1
    EAT_COW = 2
    COLLECT_SAPLING = 3
    COLLECT_DRINK = 4
    COLLECT_FOOD = 5
    MAKE_WOOD_PICKAXE = 6
    MAKE_WOOD_SWORD = 7
    PLACE_PLANT = 8
    DEFEAT_ZOMBIE = 9
    COLLECT_STONE = 10
    PLACE_STONE = 11
    EAT_PLANT = 12
    DEFEAT_SKELETON = 13
    MAKE_STONE_PICKAXE = 14
    MAKE_STONE_SWORD = 15
    WAKE_UP = 16
    PLACE_FURNACE = 17
    COLLECT_COAL = 18
    COLLECT_IRON = 19
    COLLECT_DIAMOND = 20
    MAKE_IRON_PICKAXE = 21
    MAKE_IRON_SWORD = 22

    MAKE_ARROW = 23
    MAKE_TORCH = 24
    PLACE_TORCH = 25

    COLLECT_SAPPHIRE = 55
    COLLECT_RUBY = 60
    MAKE_DIAMOND_PICKAXE = 61
    MAKE_DIAMOND_SWORD = 26
    MAKE_IRON_ARMOUR = 27
    MAKE_DIAMOND_ARMOUR = 28

    ENTER_GNOMISH_MINES = 29
    ENTER_DUNGEON = 30
    ENTER_SEWERS = 31
    ENTER_VAULT = 32
    ENTER_TROLL_MINES = 33
    ENTER_FIRE_REALM = 34
    ENTER_ICE_REALM = 35
    ENTER_GRAVEYARD = 36

    DEFEAT_GNOME_WARRIOR = 37
    DEFEAT_GNOME_ARCHER = 38
    DEFEAT_ORC_SOLIDER = 39
    DEFEAT_ORC_MAGE = 40
    DEFEAT_LIZARD = 41
    DEFEAT_KOBOLD = 42
    DEFEAT_KNIGHT = 66
    DEFEAT_ARCHER = 67
    DEFEAT_TROLL = 43
    DEFEAT_DEEP_THING = 44
    DEFEAT_PIGMAN = 45
    DEFEAT_FIRE_ELEMENTAL = 46
    DEFEAT_FROST_TROLL = 47
    DEFEAT_ICE_ELEMENTAL = 48
    DAMAGE_NECROMANCER = 49
    DEFEAT_NECROMANCER = 50

    EAT_BAT = 51
    EAT_SNAIL = 52

    FIND_BOW = 53
    FIRE_BOW = 54

    LEARN_SPELL = 56
    CAST_SPELL = 57

    OPEN_CHEST = 62
    DRINK_POTION = 63
    ENCHANT_SWORD = 64
    ENCHANT_ARMOUR = 65


INTERMEDIATE_ACHIEVEMENTS = [
    Achievement.COLLECT_SAPPHIRE.value,
    Achievement.COLLECT_RUBY.value,
    Achievement.MAKE_DIAMOND_PICKAXE.value,
    Achievement.MAKE_DIAMOND_SWORD.value,
    Achievement.MAKE_IRON_ARMOUR.value,
    Achievement.MAKE_DIAMOND_ARMOUR.value,
    Achievement.ENTER_GNOMISH_MINES.value,
    Achievement.ENTER_DUNGEON.value,
    Achievement.DEFEAT_GNOME_WARRIOR.value,
    Achievement.DEFEAT_GNOME_ARCHER.value,
    Achievement.DEFEAT_ORC_SOLIDER.value,
    Achievement.DEFEAT_ORC_MAGE.value,
    Achievement.EAT_BAT.value,
    Achievement.EAT_SNAIL.value,
    Achievement.FIND_BOW.value,
    Achievement.FIRE_BOW.value,
    Achievement.OPEN_CHEST.value,
    Achievement.DRINK_POTION.value,
]


VERY_ADVANCED_ACHIEVEMENTS = [
    Achievement.ENTER_FIRE_REALM.value,
    Achievement.ENTER_ICE_REALM.value,
    Achievement.ENTER_GRAVEYARD.value,
    Achievement.DEFEAT_PIGMAN.value,
    Achievement.DEFEAT_FIRE_ELEMENTAL.value,
    Achievement.DEFEAT_FROST_TROLL.value,
    Achievement.DEFEAT_ICE_ELEMENTAL.value,
    Achievement.DAMAGE_NECROMANCER.value,
    Achievement.DEFEAT_NECROMANCER.value,
]


def achievement_mapping(achievement_value):
    if achievement_value <= 25:
        return 1
    elif achievement_value in INTERMEDIATE_ACHIEVEMENTS:
        return 3
    elif achievement_value in VERY_ADVANCED_ACHIEVEMENTS:
        return 8
    else:
        return 5


ACHIEVEMENT_REWARD_MAP = jnp.array(
    [achievement_mapping(i) for i in range(len(Achievement))]
)


LEVEL_ACHIEVEMENT_MAP = jnp.array(
    [
        0,
        Achievement.ENTER_DUNGEON.value,
        Achievement.ENTER_GNOMISH_MINES.value,
        Achievement.ENTER_SEWERS.value,
        Achievement.ENTER_VAULT.value,
        Achievement.ENTER_TROLL_MINES.value,
        Achievement.ENTER_FIRE_REALM.value,
        Achievement.ENTER_ICE_REALM.value,
        Achievement.ENTER_GRAVEYARD.value,
    ]
)

MOB_ACHIEVEMENT_MAP = jnp.array(
    [
        # Passive
        [
            Achievement.EAT_COW.value,
            Achievement.EAT_BAT.value,
            Achievement.EAT_SNAIL.value,
            0,
            0,
            0,
            0,
            0,
        ],
        # Melee
        [
            Achievement.DEFEAT_ZOMBIE.value,
            Achievement.DEFEAT_GNOME_WARRIOR.value,
            Achievement.DEFEAT_ORC_SOLIDER.value,
            Achievement.DEFEAT_LIZARD.value,
            Achievement.DEFEAT_KNIGHT.value,
            Achievement.DEFEAT_TROLL.value,
            Achievement.DEFEAT_PIGMAN.value,
            Achievement.DEFEAT_FROST_TROLL.value,
        ],
        # Ranged
        [
            Achievement.DEFEAT_SKELETON.value,
            Achievement.DEFEAT_GNOME_ARCHER.value,
            Achievement.DEFEAT_ORC_MAGE.value,
            Achievement.DEFEAT_KOBOLD.value,
            Achievement.DEFEAT_ARCHER.value,
            Achievement.DEFEAT_DEEP_THING.value,
            Achievement.DEFEAT_FIRE_ELEMENTAL.value,
            Achievement.DEFEAT_ICE_ELEMENTAL.value,
        ],
    ]
)

# PRE-COMPUTATION
TORCH_LIGHT_MAP = get_distance_map(jnp.array([4, 4]), (9, 9))
TORCH_LIGHT_MAP /= 5.0
TORCH_LIGHT_MAP = jnp.clip(1 - TORCH_LIGHT_MAP, 0.0, 1.0)


# TEXTURES
@struct.dataclass
class PlayerSpecificTextures:
    player_textures: jnp.ndarray
    player_icon_textures: jnp.ndarray
    chest_textures: jnp.ndarray


def load_texture(filename, block_pixel_size):
    filename = os.path.join(pathlib.Path(__file__).parent.resolve(), "assets", filename)
    img = iio.imread(filename)
    jnp_img = jnp.array(img).astype(int)
    assert jnp_img.shape[:2] == (16, 16)

    if jnp_img.shape[2] == 4:
        jnp_img = jnp_img.at[:, :, 3].set(jnp_img[:, :, 3] // 255)

    if block_pixel_size != 16:
        img = np.array(jnp_img, dtype=np.uint8)
        image = Image.fromarray(img)
        image = image.resize(
            (block_pixel_size, block_pixel_size), resample=Image.NEAREST
        )
        jnp_img = jnp.array(image, dtype=jnp.int32)

    return jnp_img


def apply_alpha(texture):
    return texture[:, :, :3] * jnp.repeat(
        jnp.expand_dims(texture[:, :, 3], axis=-1), 3, axis=-1
    )

def load_player_specific_textures(texture_set, player_count) -> PlayerSpecificTextures:
    color_palette = (jnp.array(husl_palette(player_count, h=0.5, l=0.5)) * 255).astype(jnp.uint32)
    return PlayerSpecificTextures(
        player_textures=load_multiplayer_textures(
            texture_set["player_textures"], 
            color_palette, 
            player_count
        ),
        player_icon_textures=load_multiplayer_textures(
            texture_set["player_icon_textures"], 
            color_palette, 
            player_count
        )[:, :, :, :, :3],
        chest_textures=load_colored_block_textures(
            texture_set["full_map_block_textures"][BlockType.CHEST.value], 
            color_palette, 
            player_count
        )
    )

def load_multiplayer_textures(base_textures, color_palette, player_count):
    color_palette = jnp.concatenate([color_palette, jnp.ones((player_count, 1))], axis=-1)
    colors_broadcasted = color_palette[:, None, None, None, :]
    multiplayer_textures = base_textures[None, :].repeat(player_count, 0)
    mask = (multiplayer_textures == jnp.array([0, 0, 0, 1])).all(axis=-1)[..., None]
    multiplayer_textures_colored = jnp.where(mask, colors_broadcasted, multiplayer_textures)
    return multiplayer_textures_colored


def load_colored_block_textures(base_textures, color_palette, player_count):
    colors_broadcasted = color_palette[:, None, None, :]
    multiplayer_textures = base_textures[None, :].repeat(player_count, 0)
    mask = (multiplayer_textures == jnp.array([0, 0, 0])).all(axis=-1)[..., None]
    multiplayer_textures_colored = jnp.where(mask, colors_broadcasted, multiplayer_textures)
    return multiplayer_textures_colored


def load_mob_texture_set(filenames, block_pixel_size):
    textures = np.zeros((len(filenames), block_pixel_size, block_pixel_size, 3))
    texture_alphas = np.zeros((len(filenames), block_pixel_size, block_pixel_size, 3))

    for file_index, filename in enumerate(filenames):
        rgba_img = jnp.array(load_texture(filename, block_pixel_size))
        texture = apply_alpha(rgba_img)
        texture_alpha = np.repeat(
            np.expand_dims(rgba_img[:, :, 3], axis=-1), repeats=3, axis=2
        )

        textures[file_index] = texture
        texture_alphas[file_index] = texture_alpha

    return jnp.array(textures), jnp.array(texture_alphas)

def load_request_message_textures(block_pixel_size):
    icon_pixel_size = int(block_pixel_size * 0.6)
    start_loc_x = (block_pixel_size - icon_pixel_size) // 2
    start_loc_y = (block_pixel_size - icon_pixel_size) // 3
    message_bubble_texture = load_texture("message_bubble.png", block_pixel_size)

    def _overlay_item(icon_texture):
        combined_message_texture = message_bubble_texture
        
        # Only for areas where the icon is not transparent overlay the icon
        if icon_texture.shape[-1] == 4:
            original_slice = combined_message_texture[
                start_loc_y:start_loc_y + icon_pixel_size, 
                start_loc_x:start_loc_x + icon_pixel_size, 
                :3
            ]
            updated_slice = jnp.where(
                (icon_texture[:, :, 3] == 1)[:, :, None], 
                icon_texture[:, :, :3], 
                original_slice
            )
        else:
            updated_slice = icon_texture

        combined_message_texture = combined_message_texture.at[
            start_loc_y:start_loc_y + icon_pixel_size, 
            start_loc_x:start_loc_x + icon_pixel_size, 
            :3
        ].set(updated_slice)
        return combined_message_texture
    
    item_name_list = [
        "food.png",
        "drink.png",
        "wood.png",
        "stone.png",
        "iron.png",
        "coal.png",
        "diamond.png",
        "ruby.png",
        "sapphire.png",
    ]
    return jnp.array([
        _overlay_item(load_texture(f, icon_pixel_size))
        for f in item_name_list
    ])


def load_all_textures(block_pixel_size):
    small_block_pixel_size = int(block_pixel_size * 0.8)

    # Blocks
    block_texture_names = [
        "debug_tile.png",
        "debug_tile.png",
        "grass.png",
        "water.png",
        "stone.png",
        "tree.png",
        "wood.png",
        "path.png",
        "coal.png",
        "iron.png",
        "diamond.png",
        "table.png",
        "furnace.png",
        "sand.png",
        "lava.png",
        "plant_on_grass.png",
        "ripe_plant_on_grass.png",
        "wall2.png",
        "debug_tile.png",
        "wall_moss.png",
        "stalagmite.png",
        "sapphire.png",
        "ruby.png",
        "chest.png",
        "fountain.png",
        "fire_grass.png",
        "ice_grass.png",
        "gravel.png",
        "fire_tree.png",
        "ice_shrub.png",
        "enchantment_table_fire.png",
        "enchantment_table_ice.png",
        "necromancer.png",
        "grave.png",
        "grave2.png",
        "grave3.png",
        "necromancer_vulnerable.png",
    ]

    block_textures = jnp.array(
        [
            load_texture(fname, block_pixel_size)[:, :, :3]
            for fname in block_texture_names
        ]
    )

    # Manually set some textures
    block_textures = block_textures.at[BlockType.OUT_OF_BOUNDS.value].set(
        jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32) * 128
    )
    block_textures = block_textures.at[BlockType.DARKNESS.value].set(
        jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)
    )

    smaller_block_textures = jnp.array(
        [
            load_texture(fname, small_block_pixel_size)[:, :, :3]
            for fname in block_texture_names
        ]
    )

    full_map_block_textures = jnp.array(
        [jnp.tile(block_textures[block.value], (*OBS_DIM, 1)) for block in BlockType]
    )

    # Items (torches, ladders)
    item_texture_names = [
        "debug.png",
        "torch_in_inventory.png",
        "ladder_down.png",
        "ladder_up.png",
        "ladder_down_blocked.png",
    ]

    item_textures = jnp.array(
        [load_texture(fname, block_pixel_size) for fname in item_texture_names]
    )
    full_map_item_textures = jnp.array(
        [jnp.tile(item_textures[item.value], (*OBS_DIM, 1)) for item in ItemType]
    )

    # Player
    pad_pixels = (
        (OBS_DIM[0] // 2) * block_pixel_size,
        (OBS_DIM[1] // 2) * block_pixel_size,
    )

    player_textures = jnp.array(
        [
            load_texture("player-left.png", block_pixel_size),
            load_texture("player-right.png", block_pixel_size),
            load_texture("player-up.png", block_pixel_size),
            load_texture("player-down.png", block_pixel_size),
            load_texture("player-sleep.png", block_pixel_size),
            load_texture("player-dead.png", block_pixel_size),
        ]
    )

    player_icon_textures = jnp.array(
        [
            load_texture("player.png", small_block_pixel_size),
            load_texture("player-dead.png", small_block_pixel_size),
        ]
    )

    full_map_player_textures_rgba = [
        jnp.pad(
            player_texture,
            ((pad_pixels[0], pad_pixels[0]), (pad_pixels[1], pad_pixels[1]), (0, 0)),
        )
        for player_texture in player_textures
    ]

    full_map_player_textures = jnp.array(
        [player_texture[:, :, :3] for player_texture in full_map_player_textures_rgba]
    )

    full_map_player_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(player_texture[:, :, 3], axis=-1), repeats=3, axis=2
            )
            for player_texture in full_map_player_textures_rgba
        ]
    )

    # Teammate directions
    def _generate_all_direction_textures(horizontal_texture_base, diagonal_texture_base):
        right = horizontal_texture_base
        up = jnp.rot90(right, k=1)
        left = jnp.rot90(right, k=2)
        down = jnp.rot90(right, k=3)
        top_right = diagonal_texture_base
        top_left = jnp.rot90(top_right, k=1)
        bottom_left = jnp.rot90(top_right, k=2)
        bottom_right = jnp.rot90(top_right, k=3)
        return jnp.array([
            [top_left, up, top_right],
            [left, left, right],
            [bottom_left, down, bottom_right],
        ])

    direction_texture_base = load_texture("pointer-right.png", small_block_pixel_size)
    direction_diagonal_texture_base = load_texture("pointer-top-right.png", small_block_pixel_size)
    direction_textures = _generate_all_direction_textures(direction_texture_base, direction_diagonal_texture_base)

    # inventory

    empty_texture = jnp.zeros((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)
    smaller_empty_texture = jnp.zeros(
        (small_block_pixel_size, small_block_pixel_size, 3), dtype=jnp.int32
    )

    ones_texture = jnp.ones((block_pixel_size, block_pixel_size, 3), dtype=jnp.int32)

    number_size = int(block_pixel_size * 0.4)

    number_textures_rgba = [
        jnp.zeros((number_size, number_size, 3), dtype=jnp.int32),
        load_texture("1.png", number_size),
        load_texture("2.png", number_size),
        load_texture("3.png", number_size),
        load_texture("4.png", number_size),
        load_texture("5.png", number_size),
        load_texture("6.png", number_size),
        load_texture("7.png", number_size),
        load_texture("8.png", number_size),
        load_texture("9.png", number_size),
    ]

    number_textures = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in number_textures_rgba
        ]
    )

    number_textures_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2
            )
            for number_texture in number_textures_rgba
        ]
    )

    number_textures_with_zero_rgba = [
        load_texture("0.png", number_size),
        load_texture("1.png", number_size),
        load_texture("2.png", number_size),
        load_texture("3.png", number_size),
        load_texture("4.png", number_size),
        load_texture("5.png", number_size),
        load_texture("6.png", number_size),
        load_texture("7.png", number_size),
        load_texture("8.png", number_size),
        load_texture("9.png", number_size),
    ]

    number_textures_with_zero = jnp.array(
        [
            number_texture[:, :, :3]
            * jnp.repeat(jnp.expand_dims(number_texture[:, :, 3], axis=-1), 3, axis=-1)
            for number_texture in number_textures_with_zero_rgba
        ]
    )

    number_textures_with_zero_alpha = jnp.array(
        [
            jnp.repeat(
                jnp.expand_dims(number_texture[:, :, 3], axis=-1), repeats=3, axis=2
            )
            for number_texture in number_textures_with_zero_rgba
        ]
    )

    health_texture = jnp.array(
        load_texture("health.png", small_block_pixel_size)[:, :, :3]
    )
    hunger_texture = jnp.array(
        load_texture("food.png", small_block_pixel_size)[:, :, :3]
    )
    thirst_texture = jnp.array(
        load_texture("drink.png", small_block_pixel_size)[:, :, :3]
    )
    energy_texture = jnp.array(
        load_texture("energy.png", small_block_pixel_size)[:, :, :3]
    )
    mana_texture = jnp.array(load_texture("mana.png", small_block_pixel_size)[:, :, :3])

    pickaxe_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in [
                "debug.png",
                "wood_pickaxe.png",
                "stone_pickaxe.png",
                "iron_pickaxe.png",
                "diamond_pickaxe.png",
            ]
        ]
    )
    pickaxe_textures = pickaxe_textures.at[0].set(smaller_empty_texture)

    sword_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in [
                "debug.png",
                "wood_sword.png",
                "stone_sword.png",
                "iron_sword.png",
                "diamond_sword.png",
            ]
        ]
    )
    sword_textures = sword_textures.at[0].set(smaller_empty_texture)

    iron_armour_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in [
                "iron_helmet.png",
                "iron_chestplate.png",
                "iron_pants.png",
                "iron_boots.png",
            ]
        ]
    )
    diamond_armour_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in [
                "diamond_helmet.png",
                "diamond_chestplate.png",
                "diamond_pants.png",
                "diamond_boots.png",
            ]
        ]
    )
    empty_armour_textures = jnp.stack(
        [
            smaller_empty_texture,
            smaller_empty_texture,
            smaller_empty_texture,
            smaller_empty_texture,
        ],
        axis=0,
    )

    armour_textures = jnp.stack(
        [empty_armour_textures, iron_armour_textures, diamond_armour_textures], axis=0
    )

    bow_texture = load_texture("bow.png", small_block_pixel_size)[:, :, :3]
    bow_textures = jnp.stack([smaller_empty_texture, bow_texture], axis=0)
    player_projectile_textures = jnp.array(
        [
            apply_alpha(load_texture(filename, small_block_pixel_size))
            for filename in ["arrow-up.png", "debug.png", "fireball.png", "iceball.png"]
        ]
    )

    sapling_texture = jnp.array(
        load_texture("sapling.png", small_block_pixel_size)[:, :, :3]
    )

    torch_inv_texture = jnp.array(
        load_texture("torch_in_inventory.png", small_block_pixel_size)[:, :, :3]
    )

    # entities
    melee_mob_textures, melee_mob_texture_alphas = load_mob_texture_set(
        [
            "zombie.png",
            "gnome_warrior.png",
            "orc_soldier.png",
            "lizard.png",
            "knight.png",
            "troll.png",
            "pigman.png",
            "frost_troll.png",
        ],
        block_pixel_size,
    )
    passive_mob_textures, passive_mob_texture_alphas = load_mob_texture_set(
        ["cow.png", "bat.png", "snail.png"], block_pixel_size
    )
    ranged_mob_textures, ranged_mob_texture_alphas = load_mob_texture_set(
        [
            "skeleton.png",
            "gnome_archer.png",
            "orc_mage.png",
            "kobold.png",
            "knight_archer.png",
            "deep_thing.png",
            "fire_elemental.png",
            "ice_elemental.png",
        ],
        block_pixel_size,
    )
    projectile_textures, projectile_texture_alphas = load_mob_texture_set(
        [
            "arrow-up.png",
            "dagger.png",
            "fireball.png",
            "iceball.png",
            "arrow-up.png",
            "slimeball.png",
            "fireball.png",
            "iceball.png",
        ],
        block_pixel_size,
    )

    night_texture = (
        jnp.array([[[0, 16, 64]]])
        .repeat(OBS_DIM[0] * block_pixel_size, axis=0)
        .repeat(OBS_DIM[1] * block_pixel_size, axis=1)
    )

    night_noise_intensity_texture = jnp.array(
        [
            [
                jnp.sqrt(
                    (x - (OBS_DIM[0] * block_pixel_size // 2)) ** 2
                    + (y - (OBS_DIM[1] * block_pixel_size // 2)) ** 2
                )
                for y in range(OBS_DIM[1] * block_pixel_size)
            ]
            for x in range(OBS_DIM[0] * block_pixel_size)
        ]
    )
    night_noise_intensity_texture = (
        night_noise_intensity_texture / night_noise_intensity_texture.max()
    )

    night_noise_intensity_texture = jnp.expand_dims(
        night_noise_intensity_texture, axis=-1
    ).repeat(3, axis=-1)

    potion_textures = jnp.array(
        [
            load_texture("potion_red.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_green.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_blue.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_pink.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_cyan.png", small_block_pixel_size)[:, :, :3],
            load_texture("potion_yellow.png", small_block_pixel_size)[:, :, :3],
        ]
    )

    book_texture = load_texture("book.png", small_block_pixel_size)[:, :, :3]

    fireball_inv_texture = load_texture("fireball.png", small_block_pixel_size)[
        :, :, :3
    ]
    iceball_inv_texture = load_texture("iceball.png", small_block_pixel_size)[:, :, :3]
    heal_inv_texture = load_texture("heal_cross.png", small_block_pixel_size)[:, :, :3]

    # Attributes
    xp_texture = load_texture("xp.png", small_block_pixel_size)[:, :, :3]
    dex_texture = load_texture("dexterity.png", small_block_pixel_size)[:, :, :3]
    str_texture = load_texture("strength.png", small_block_pixel_size)[:, :, :3]
    int_texture = load_texture("intelligence.png", small_block_pixel_size)[:, :, :3]

    # Specializations
    forager_texture = load_texture("forager.png", small_block_pixel_size)[:, :, :3]
    warrior_texture = load_texture("warrior.png", small_block_pixel_size)[:, :, :3]
    miner_texture = load_texture("miner.png", small_block_pixel_size)[:, :, :3]

    armour_enchantment_textures = jnp.array(
        [
            [
                jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
                jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
                jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
                jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
            ],
            [
                load_texture("helmet_fire_enchantment.png", small_block_pixel_size),
                load_texture("chestplate_fire_enchantment.png", small_block_pixel_size),
                load_texture("pants_fire_enchantment.png", small_block_pixel_size),
                load_texture("boots_fire_enchantment.png", small_block_pixel_size),
            ],
            [
                load_texture("helmet_ice_enchantment.png", small_block_pixel_size),
                load_texture("chestplate_ice_enchantment.png", small_block_pixel_size),
                load_texture("pants_ice_enchantment.png", small_block_pixel_size),
                load_texture("boots_ice_enchantment.png", small_block_pixel_size),
            ],
        ]
    )

    sword_enchantment_textures = jnp.array(
        [
            jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
            load_texture("sword_fire_enchantment.png", small_block_pixel_size),
            load_texture("sword_ice_enchantment.png", small_block_pixel_size),
        ]
    )

    arrow_enchantment_textures = jnp.array(
        [
            jnp.zeros((small_block_pixel_size, small_block_pixel_size, 4)),
            load_texture("arrow_fire_enchantment.png", small_block_pixel_size),
            load_texture("arrow_ice_enchantment.png", small_block_pixel_size),
        ]
    )

    request_message_textures = load_request_message_textures(small_block_pixel_size)

    return {
        "block_textures": block_textures,
        "smaller_block_textures": smaller_block_textures,
        "full_map_block_textures": full_map_block_textures,
        "full_map_item_textures": full_map_item_textures,
        "player_textures": player_textures,
        "full_map_player_textures": full_map_player_textures,
        "full_map_player_textures_alpha": full_map_player_textures_alpha,
        "player_icon_textures": player_icon_textures,
        "empty_texture": empty_texture,
        "smaller_empty_texture": smaller_empty_texture,
        "ones_texture": ones_texture,
        "number_textures": number_textures,
        "number_textures_alpha": number_textures_alpha,
        "number_textures_with_zero": number_textures_with_zero,
        "number_textures_alpha_with_zero": number_textures_with_zero_alpha,
        "health_texture": health_texture,
        "hunger_texture": hunger_texture,
        "thirst_texture": thirst_texture,
        "energy_texture": energy_texture,
        "mana_texture": mana_texture,
        "pickaxe_textures": pickaxe_textures,
        "sword_textures": sword_textures,
        "sapling_texture": sapling_texture,
        "night_texture": night_texture,
        "night_noise_intensity_texture": night_noise_intensity_texture,
        "melee_mob_textures": melee_mob_textures,
        "melee_mob_texture_alphas": melee_mob_texture_alphas,
        "passive_mob_textures": passive_mob_textures,
        "passive_mob_texture_alphas": passive_mob_texture_alphas,
        "direction_textures": direction_textures,
        "ranged_mob_textures": ranged_mob_textures,
        "ranged_mob_texture_alphas": ranged_mob_texture_alphas,
        "projectile_textures": projectile_textures,
        "projectile_texture_alphas": projectile_texture_alphas,
        "armour_textures": armour_textures,
        "bow_textures": bow_textures,
        "player_projectile_textures": player_projectile_textures,
        "torch_inv_texture": torch_inv_texture,
        "potion_textures": potion_textures,
        "book_texture": book_texture,
        "fireball_inv_texture": fireball_inv_texture,
        "iceball_inv_texture": iceball_inv_texture,
        "heal_inv_texture": heal_inv_texture,
        "armour_enchantment_textures": armour_enchantment_textures,
        "sword_enchantment_textures": sword_enchantment_textures,
        "arrow_enchantment_textures": arrow_enchantment_textures,
        "xp_texture": xp_texture,
        "dex_texture": dex_texture,
        "str_texture": str_texture,
        "int_texture": int_texture,
        "forager_texture": forager_texture,
        "warrior_texture": warrior_texture,
        "miner_texture": miner_texture,
        "request_message_textures": request_message_textures,
    }


if os.path.exists(TEXTURE_CACHE_FILE) and not os.environ.get(
    "CRAFTAX_RELOAD_TEXTURES", False
):
    print("Loading textures from cache")
    TEXTURES = load_compressed_pickle(TEXTURE_CACHE_FILE)
else:
    print("Processing textures")
    TEXTURES = {
        BLOCK_PIXEL_SIZE_AGENT: load_all_textures(BLOCK_PIXEL_SIZE_AGENT),
        BLOCK_PIXEL_SIZE_IMG: load_all_textures(BLOCK_PIXEL_SIZE_IMG),
        BLOCK_PIXEL_SIZE_HUMAN: load_all_textures(BLOCK_PIXEL_SIZE_HUMAN),
    }
    save_compressed_pickle(TEXTURE_CACHE_FILE, TEXTURES)
    print("Textures saved to cache")
