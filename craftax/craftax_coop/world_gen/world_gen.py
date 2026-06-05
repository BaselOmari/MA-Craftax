import jax
import jax.scipy as jsp

from craftax_coop.constants import *
from craftax_coop.game_logic import calculate_light_level
from craftax_coop.util.maths_utils import get_all_players_distance_map
from craftax_coop.craftax_state import EnvState, Inventory, Mobs
from craftax_coop.util.game_logic_utils import get_ladder_positions
from craftax_coop.util.noise import generate_fractal_noise_2d
from craftax_coop.world_gen.world_gen_configs import (
    ALL_DUNGEON_CONFIGS,
    ALL_SMOOTHGEN_CONFIGS,
)


def get_new_empty_inventory(player_count):
    return Inventory(
        wood=jnp.full((player_count,), 0, dtype=jnp.int32),
        stone=jnp.full((player_count,), 0, dtype=jnp.int32),
        coal=jnp.full((player_count,), 0, dtype=jnp.int32),
        iron=jnp.full((player_count,), 0, dtype=jnp.int32),
        diamond=jnp.full((player_count,), 0, dtype=jnp.int32),
        sapling=jnp.full((player_count,), 0, dtype=jnp.int32),
        pickaxe=jnp.full((player_count,), 0, dtype=jnp.int32),
        sword=jnp.full((player_count,), 0, dtype=jnp.int32),
        bow=jnp.full((player_count,), 0, dtype=jnp.int32),
        arrows=jnp.full((player_count,), 0, dtype=jnp.int32),
        torches=jnp.full((player_count,), 0, dtype=jnp.int32),
        ruby=jnp.full((player_count,), 0, dtype=jnp.int32),
        sapphire=jnp.full((player_count,), 0, dtype=jnp.int32),
        books=jnp.full((player_count,), 0, dtype=jnp.int32),
        potions=jnp.full((player_count, 6), 0, dtype=jnp.int32),
        armour=jnp.full((player_count, 4), 0, dtype=jnp.int32),
    )


def get_new_full_inventory(player_count):
    return Inventory(
        wood=jnp.full((player_count,), 99, dtype=jnp.int32),
        stone=jnp.full((player_count,), 99, dtype=jnp.int32),
        coal=jnp.full((player_count,), 99, dtype=jnp.int32),
        iron=jnp.full((player_count,), 99, dtype=jnp.int32),
        diamond=jnp.full((player_count,), 99, dtype=jnp.int32),
        sapling=jnp.full((player_count,), 99, dtype=jnp.int32),
        pickaxe=jnp.full((player_count,), 4, dtype=jnp.int32),
        sword=jnp.full((player_count,), 4, dtype=jnp.int32),
        bow=jnp.full((player_count,), 1, dtype=jnp.int32),
        arrows=jnp.full((player_count,), 99, dtype=jnp.int32),
        torches=jnp.full((player_count,), 99, dtype=jnp.int32),
        ruby=jnp.full((player_count,), 99, dtype=jnp.int32),
        sapphire=jnp.full((player_count,), 99, dtype=jnp.int32),
        books=jnp.full((player_count,), 99, dtype=jnp.int32),
        potions=jnp.full((player_count, 6), 99, dtype=jnp.int32),
        armour=jnp.full((player_count, 4), 2, dtype=jnp.int32),
    )


def generate_dungeon(rng, static_params, config):
    chunk_size = 16
    world_chunk_width = static_params.map_size[0] // chunk_size
    world_chunk_height = static_params.map_size[1] // chunk_size
    room_occupancy_chunks = jnp.ones(world_chunk_width * world_chunk_height)

    rng, _rng, __rng = jax.random.split(rng, 3)
    room_sizes = jax.random.randint(
        __rng, shape=(NUM_ROOMS, 2), minval=MIN_ROOM_SIZE, maxval=MAX_ROOM_SIZE
    )

    map = jnp.ones(static_params.map_size, dtype=jnp.int32) * BlockType.WALL.value
    padded_map = jnp.pad(map, MAX_ROOM_SIZE, constant_values=0)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)
    padded_item_map = jnp.pad(item_map, MAX_ROOM_SIZE, constant_values=0)

    def _add_room(carry, room_index):
        block_map, item_map, room_occupancy_chunks, rng = carry

        rng, _rng = jax.random.split(rng)
        room_chunk = jax.random.choice(
            _rng,
            jnp.arange(world_chunk_width * world_chunk_height),
            p=room_occupancy_chunks,
        )
        room_occupancy_chunks = room_occupancy_chunks.at[room_chunk].set(0)

        room_position = jnp.array(
            [
                (room_chunk % world_chunk_height) * chunk_size,
                (room_chunk // world_chunk_height) * chunk_size,
            ]
        ) + jnp.array([MAX_ROOM_SIZE, MAX_ROOM_SIZE])
        rng, _rng = jax.random.split(rng)
        room_position += jax.random.randint(
            _rng, (2,), minval=0, maxval=chunk_size - MIN_ROOM_SIZE
        )

        slice = jax.lax.dynamic_slice(
            block_map, room_position, (MAX_ROOM_SIZE, MAX_ROOM_SIZE)
        )
        xs = jnp.expand_dims(jnp.arange(MAX_ROOM_SIZE), axis=-1).repeat(
            MAX_ROOM_SIZE, axis=-1
        )
        ys = jnp.expand_dims(jnp.arange(MAX_ROOM_SIZE), axis=0).repeat(
            MAX_ROOM_SIZE, axis=0
        )

        room_mask = jnp.logical_and(
            xs < room_sizes[room_index, 0], ys < room_sizes[room_index, 1]
        )

        slice = room_mask * BlockType.PATH.value + (1 - room_mask) * slice

        block_map = jax.lax.dynamic_update_slice(
            block_map,
            slice,
            room_position,
        )

        # Torches in corner
        item_map = item_map.at[room_position[0], room_position[1]].set(
            ItemType.TORCH.value
        )
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1, room_position[1]
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0], room_position[1] + room_sizes[room_index, 1] - 1
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1,
            room_position[1] + room_sizes[room_index, 1] - 1,
        ].set(ItemType.TORCH.value)

        # Chest
        rng, _rng = jax.random.split(rng)
        chest_position = jax.random.randint(
            _rng,
            shape=(static_params.player_count, 2),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        block_map = block_map.at[
            room_position[0] + chest_position[:, 0], room_position[1] + chest_position[:, 1]
        ].set(BlockType.CHEST.value)

        # Fountain
        rng, _rng, __rng = jax.random.split(rng, 3)
        fountain_position = jax.random.randint(
            _rng,
            shape=(2,),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        room_has_fountain = jax.random.uniform(__rng) < config.fountain_probability
        fountain_block = (
            room_has_fountain * config.fountain_block
            + (1 - room_has_fountain)
            * block_map[
                room_position[0] + fountain_position[0],
                room_position[1] + fountain_position[1],
            ]
        )
        block_map = block_map.at[
            room_position[0] + fountain_position[0],
            room_position[1] + fountain_position[1],
        ].set(fountain_block)

        return (block_map, item_map, room_occupancy_chunks, rng), room_position

    rng, _rng = jax.random.split(rng)
    (padded_map, padded_item_map, _, _), room_positions = jax.lax.scan(
        _add_room,
        (padded_map, padded_item_map, room_occupancy_chunks, _rng),
        jnp.arange(NUM_ROOMS),
    )

    corridor_width = static_params.corridor_width

    def _add_path(carry, path_index):
        cmap, included_rooms_mask, rng = carry

        path_source = room_positions[path_index]

        rng, _rng = jax.random.split(rng)
        sink_index = jax.random.choice(
            _rng, jnp.arange(NUM_ROOMS), p=included_rooms_mask
        )
        path_sink = room_positions[sink_index]

        # Horizontal component
        map_height, map_width = cmap.shape
        horizontal_rows = jax.lax.dynamic_slice(
            cmap,
            (path_source[0], 0),
            (corridor_width, map_width),
        )
        path_indexes = jnp.arange(map_width)
        path_indexes = path_indexes - path_source[1]
        horizontal_distance = path_sink[1] - path_source[1]
        path_indexes = path_indexes * jnp.sign(horizontal_distance)

        horizontal_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(
            horizontal_mask, jnp.sign(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(
            horizontal_mask[None, :], horizontal_rows == BlockType.WALL.value
        )

        new_rows = jnp.where(
            horizontal_mask,
            BlockType.PATH.value,
            horizontal_rows,
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            new_rows,
            (path_source[0], 0),
        )

        # Vertical component
        vertical_cols = jax.lax.dynamic_slice(
            cmap,
            (0, path_sink[1]),
            (map_height, corridor_width),
        )
        path_indexes = jnp.arange(map_height)
        path_indexes = path_indexes - path_source[0]
        vertical_distance = path_sink[0] - path_source[0]
        path_indexes = path_indexes * jnp.sign(vertical_distance)

        vertical_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(vertical_distance)
        )
        vertical_mask = jnp.logical_and(vertical_mask, jnp.sign(vertical_distance))

        vertical_mask = jnp.logical_and(
            vertical_mask[:, None], vertical_cols == BlockType.WALL.value
        )

        new_cols = jnp.where(
            vertical_mask,
            BlockType.PATH.value,
            vertical_cols,
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            new_cols,
            (0, path_sink[1]),
        )

        rng, _rng = jax.random.split(rng)
        included_rooms_mask = included_rooms_mask.at[path_index].set(True)
        return (cmap, included_rooms_mask, _rng), None

    rng, _rng = jax.random.split(rng)
    included_rooms_mask = jnp.zeros(NUM_ROOMS, dtype=bool).at[-1].set(True)
    (
        (padded_map, _, _),
        _,
    ) = jax.lax.scan(
        _add_path, (padded_map, included_rooms_mask, _rng), jnp.arange(0, NUM_ROOMS)
    )

    # Place special block in a random room
    special_block_position = room_positions[0] + jnp.array([2, 2])
    padded_map = padded_map.at[
        special_block_position[0], special_block_position[1]
    ].set(config.special_block)

    map = padded_map[MAX_ROOM_SIZE:-MAX_ROOM_SIZE, MAX_ROOM_SIZE:-MAX_ROOM_SIZE]
    item_map = padded_item_map[
        MAX_ROOM_SIZE:-MAX_ROOM_SIZE, MAX_ROOM_SIZE:-MAX_ROOM_SIZE
    ]

    # Visual stuff
    c_path_map = map != BlockType.WALL.value
    z = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    adj_path_map = jsp.signal.convolve(c_path_map, z, mode="same")
    adj_path_map = adj_path_map > 0.5

    rng, _rng = jax.random.split(rng)
    rare_map = jax.random.choice(
        _rng,
        jnp.array([False, True]),
        static_params.map_size,
        p=jnp.array([0.9, 0.1]),
    )

    wall_map = (
        rare_map * BlockType.WALL_MOSS.value + (1 - rare_map) * BlockType.WALL.value
    )

    rare_map = jnp.logical_and(rare_map, map == BlockType.PATH.value)
    rare_map = jnp.logical_and(rare_map, item_map == ItemType.NONE.value)
    path_map = rare_map * config.rare_path_replacement_block + (1 - rare_map) * map

    is_wall_map = jnp.logical_and(map == BlockType.WALL.value, adj_path_map)
    is_darkness_map = jnp.logical_not(adj_path_map)
    is_path_map = jnp.logical_not(jnp.logical_or(is_wall_map, is_darkness_map))

    map = (
        is_path_map * path_map
        + is_wall_map * wall_map
        + is_darkness_map * BlockType.DARKNESS.value
    )

    # Place SNAIL_SPAWN tiles per room with configured probability.
    # Each selected room gets exactly one inner PATH tile converted to SNAIL_SPAWN.
    rng, _rng = jax.random.split(rng)
    room_snail_rolls = jax.random.uniform(_rng, shape=(NUM_ROOMS,))
    room_has_snail = room_snail_rolls < config.snail_spawn_room_probability
    # Unpad room positions to map coordinates
    unpadded_positions = room_positions - MAX_ROOM_SIZE

    def _mark_room_snail(carry, room_index):
        current_map, room_rng = carry
        room_rng, _room_rng = jax.random.split(room_rng)
        rp = unpadded_positions[room_index]
        rs = room_sizes[room_index]
        # Build mask for tiles inside this room (excluding border for inner tiles)
        rows = jnp.arange(static_params.map_size[0])
        cols = jnp.arange(static_params.map_size[1])
        in_room_rows = jnp.logical_and(rows >= rp[0] + 1, rows < rp[0] + rs[0] - 1)
        in_room_cols = jnp.logical_and(cols >= rp[1] + 1, cols < rp[1] + rs[1] - 1)
        in_room = in_room_rows[:, None] & in_room_cols[None, :]
        # Only convert PATH tiles without items
        is_path = current_map == BlockType.PATH.value
        no_item = item_map == ItemType.NONE.value
        valid_snail_tiles = in_room & is_path & no_item

        # Sample exactly one valid tile in this room (if any exists).
        valid_flat = valid_snail_tiles.reshape(-1).astype(jnp.float32)
        valid_count = valid_flat.sum()
        num_tiles = valid_flat.shape[0]
        probs = jnp.where(
            valid_count > 0.0,
            valid_flat / valid_count,
            jnp.full_like(valid_flat, 1.0 / num_tiles),
        )
        flat_idx = jax.random.choice(_room_rng, jnp.arange(num_tiles), p=probs)
        tile_r = flat_idx // static_params.map_size[1]
        tile_c = flat_idx % static_params.map_size[1]

        should_place = room_has_snail[room_index] & (valid_count > 0.0)
        current_val = current_map[tile_r, tile_c]
        new_val = jax.lax.select(
            should_place,
            jnp.asarray(BlockType.SNAIL_SPAWN.value, dtype=current_val.dtype),
            current_val,
        )
        current_map = current_map.at[tile_r, tile_c].set(new_val)
        return (current_map, room_rng), None

    (map, _), _ = jax.lax.scan(_mark_room_snail, (map, rng), jnp.arange(NUM_ROOMS))

    light_map = jnp.ones(static_params.map_size, dtype=jnp.float32)

    # Ladders — disabled (single-level environment, no floor changes)
    rng, _rng = jax.random.split(rng)
    ladders_down = get_ladder_positions(_rng, static_params, config, map)

    rng, _rng = jax.random.split(rng)
    ladders_up = get_ladder_positions(_rng, static_params, config, map)

    # Convert room positions from padded to unpadded coordinates
    unpadded_room_positions = room_positions - MAX_ROOM_SIZE  # (NUM_ROOMS, 2)

    return map, item_map, light_map, ladders_down, ladders_up, unpadded_room_positions, room_sizes


def generate_smoothworld(rng, static_params, player_position, config, params=None):
    if params is not None:
        fractal_noise_angles = params.fractal_noise_angles
    else:
        fractal_noise_angles = (None, None, None, None, None)

    player_proximity_map = get_all_players_distance_map(
        player_position, jnp.full(static_params.player_count, True), static_params
    )
    player_proximity_map_water = (
        player_proximity_map / config.player_proximity_map_water_strength
    )
    player_proximity_map_water = jnp.clip(
        player_proximity_map_water, 0.0, config.player_proximity_map_water_max
    )

    player_proximity_map_mountain = (
        player_proximity_map / config.player_proximity_map_mountain_strength
    )
    player_proximity_map_mountain = jnp.clip(
        player_proximity_map_mountain,
        0.0,
        config.player_proximity_map_mountain_max,
    )

    larger_res = (static_params.map_size[0] // 4, static_params.map_size[1] // 4)
    small_res = (static_params.map_size[0] // 16, static_params.map_size[1] // 16)
    x_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 2)

    rng, _rng = jax.random.split(rng)
    water = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        small_res,
        octaves=1,
        override_angles=fractal_noise_angles[0],
    )
    water = water + player_proximity_map_water - 1.0

    # Water
    rng, _rng = jax.random.split(rng)
    map = jnp.where(
        water > config.water_threshold, config.sea_block, config.default_block
    )

    sand_map = jnp.logical_and(
        water > config.sand_threshold,
        map != config.sea_block,
    )

    map = jnp.where(sand_map, config.coast_block, map)

    # Mountain vs grass
    mountain_threshold = 0.7

    rng, _rng = jax.random.split(rng)
    mountain = (
        generate_fractal_noise_2d(
            _rng,
            static_params.map_size,
            small_res,
            octaves=1,
            override_angles=fractal_noise_angles[1],
        )
        + 0.05
    )
    mountain = mountain + player_proximity_map_mountain - 1.0
    map = jnp.where(mountain > mountain_threshold, config.mountain_block, map)

    # Paths
    rng, _rng = jax.random.split(rng)
    path_x = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        x_res,
        octaves=1,
        override_angles=fractal_noise_angles[2],
    )
    path = jnp.logical_and(mountain > mountain_threshold, path_x > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    path_y = path_x.T
    path = jnp.logical_and(mountain > mountain_threshold, path_y > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    # Caves
    rng, _rng = jax.random.split(rng)
    caves = jnp.logical_and(mountain > 0.85, water > 0.4)
    map = jnp.where(caves > 0.5, config.inner_mountain_block, map)

    # Trees
    rng, _rng = jax.random.split(rng)
    tree_noise = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        larger_res,
        octaves=1,
        override_angles=fractal_noise_angles[3],
    )
    tree = (tree_noise > config.tree_threshold_perlin) * jax.random.uniform(
        rng, shape=static_params.map_size
    ) > config.tree_threshold_uniform
    tree = jnp.logical_and(tree, map == config.tree_requirement_block)
    map = jnp.where(tree, config.tree, map)

    # Ores
    def _add_ore(carry, index):
        rng, map = carry
        rng, _rng = jax.random.split(rng)
        ore_map = jnp.logical_and(
            map == config.ore_requirement_blocks[index],
            jax.random.uniform(_rng, static_params.map_size)
            < config.ore_chances[index],
        )
        map = jnp.where(ore_map, config.ores[index], map)

        return (rng, map), None

    rng, _rng = jax.random.split(rng)
    (_, map), _ = jax.lax.scan(_add_ore, (_rng, map), jnp.arange(5))

    # Lava
    lava_map = jnp.logical_and(
        mountain > 0.85,
        tree_noise > 0.7,
    )
    map = jnp.where(lava_map, config.lava, map)

    # Light map
    light_map = (
        jnp.ones(static_params.map_size, dtype=jnp.float32) * config.default_light
    )

    # Make sure player spawns on grass
    map = map.at[player_position[:, 0], player_position[:, 1]].set(config.player_spawn)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)

    rng, _rng = jax.random.split(rng)
    ladders_down = get_ladder_positions(_rng, static_params, config, map)

    item_map = item_map.at[ladders_down[:, 0], ladders_down[:, 1]].set(
        ItemType.LADDER_DOWN.value * config.ladder_down
        + map[ladders_down[:, 0], ladders_down[:, 1]] * (1 - config.ladder_down)
    )

    rng, _rng = jax.random.split(rng)
    ladders_up = get_ladder_positions(_rng, static_params, config, map)

    LIGHT_MAP_AROUND_LADDER = TORCH_LIGHT_MAP * (
        1 - config.default_light
    ) + config.default_light * jnp.ones((9, 9))

    def _set_ladder_light(light_map, ladder_position):
        out = jax.lax.dynamic_update_slice(
            light_map, LIGHT_MAP_AROUND_LADDER, ladder_position - jnp.array([4, 4])
        )
        return out, None

    light_map, _ = jax.lax.scan(_set_ladder_light, light_map, ladders_up)

    z = jnp.array([[0.2, 0.7, 0.2], [0.7, 1, 0.7], [0.2, 0.7, 0.2]]) * (
        config.lava == BlockType.LAVA.value
    )
    light_map += jsp.signal.convolve(lava_map, z, mode="same")
    light_map = jnp.clip(light_map, 0.0, 1.0)

    item_map = item_map.at[ladders_up[:, 0], ladders_up[:, 1]].set(
        ItemType.LADDER_UP.value * config.ladder_up
        + map[ladders_up[:, 0], ladders_up[:, 1]] * (1 - config.ladder_up)
    )

    return map, item_map, light_map, ladders_down, ladders_up


def generate_world(rng, params, static_params):
    # --- Phase 1: Generate all maps first (before choosing spawn) ---
    # We need a temporary player_position for smoothgen (it uses it for
    # proximity maps to push water/mountains away).  Place it at map centre;
    # the real spawn will be chosen after map generation from PATH tiles.
    map_h, map_w = static_params.map_size[0], static_params.map_size[1]
    temp_center = jnp.array([map_h // 2, map_w // 2])
    temp_player_position = jnp.tile(temp_center, (static_params.player_count, 1))

    agents_per_team = len(static_params.team_composition)
    if agents_per_team <= 0:
        raise ValueError("team_composition must contain at least one role.")
    if static_params.num_teams <= 0:
        raise ValueError("num_teams must be >= 1.")
    expected_player_count = static_params.num_teams * agents_per_team
    if static_params.player_count != expected_player_count:
        raise ValueError(
            f"player_count ({static_params.player_count}) must equal "
            f"num_teams * len(team_composition) ({expected_player_count})."
        )
    valid_specializations = {
        Specialization.FORAGER.value,
        Specialization.WARRIOR.value,
        Specialization.MINER.value,
    }
    invalid_roles = [
        role for role in static_params.team_composition
        if role not in valid_specializations
    ]
    if invalid_roles:
        raise ValueError(
            "team_composition contains invalid role ids. "
            "Allowed values are FORAGER=1, WARRIOR=2, MINER=3. "
            f"Got invalid values: {invalid_roles}"
        )

    has_non_forager_lone_room = any(
        role != Specialization.FORAGER.value
        for role in static_params.team_composition
    )
    required_spawn_rooms = (
        3 if has_non_forager_lone_room else 2
    ) * static_params.num_teams
    if required_spawn_rooms > NUM_ROOMS:
        rooms_per_team = 3 if has_non_forager_lone_room else 2
        raise ValueError(
            f"spawn layout requires {required_spawn_rooms} rooms "
            f"({rooms_per_team} per team for {static_params.num_teams} teams), "
            f"but only {NUM_ROOMS} rooms are available."
        )

    # Fix player specializations from team_composition config
    # e.g. team_composition=(1, 1, 2) with num_teams=2 -> [1, 1, 2, 1, 1, 2]
    comp = jnp.array(static_params.team_composition)
    player_specializations = jnp.tile(comp, static_params.num_teams)
    # Role-aware index of each forager within a team composition.
    # Example:
    #   [FORAGER, FORAGER, WARRIOR] -> [0, 1, 1]
    #   [WARRIOR, FORAGER, FORAGER] -> [0, 0, 1]
    forager_rank_by_slot = jnp.clip(
        jnp.cumsum((comp == Specialization.FORAGER.value).astype(jnp.int32)) - 1,
        0,
        1,
    )

    # Fix player subclasses (team assignment)
    # e.g. 6 players, 3 per team -> [0, 0, 0, 1, 1, 1]
    player_sc = jnp.arange(static_params.player_count) // agents_per_team

    # Generate smoothgens (overworld, caves, elemental levels, boss level)
    rngs = jax.random.split(rng, 7)
    rng, _rng = rngs[0], rngs[1:]
    smoothgens = jax.vmap(generate_smoothworld, in_axes=(0, None, None, 0))(
        _rng, static_params, temp_player_position, ALL_SMOOTHGEN_CONFIGS
    )

    # Generate dungeons
    rngs = jax.random.split(rng, 4)
    rng, _rng = rngs[0], rngs[1:]
    dungeon_results = jax.vmap(generate_dungeon, in_axes=(0, None, 0))(
        _rng, static_params, ALL_DUNGEON_CONFIGS
    )
    # Separate room metadata from map data
    # dungeon_results = (maps, item_maps, light_maps, ladders_down, ladders_up, room_positions, room_sizes)
    d_maps, d_item_maps, d_light_maps, d_ladders_down, d_ladders_up, dungeon_room_positions, dungeon_room_sizes = dungeon_results
    dungeons = (d_maps, d_item_maps, d_light_maps, d_ladders_down, d_ladders_up)
    # dungeon_room_positions: (3, NUM_ROOMS, 2) top-left corner of each room (unpadded)
    # dungeon_room_sizes:     (3, NUM_ROOMS, 2) (height, width) of each room

    # Returns stacked versions of the map, item_map, light_map and ladders
    # 9 elements in each of these stacks representing each of the levels.
    # Splice smoothgens and dungeons in order of levels
    map, item_map, light_map, ladders_down, ladders_up = jax.tree_util.tree_map(
        lambda x, y: jnp.stack(
            (x[0], x[1], y[0], y[1], y[2], x[2], x[3], x[4], x[5]), axis=0
        ),
        smoothgens,
        dungeons,
    )

    # --- Phase 2: Pick team spawn positions inside ROOMS on the start level ---
    START_LEVEL = 2  # First dungeon level (dungeon index 0)
    start_room_positions = dungeon_room_positions[0]  # (NUM_ROOMS, 2) top-left corners
    start_room_sizes = dungeon_room_sizes[0]          # (NUM_ROOMS, 2) (h, w)

    # Compute room centers
    room_centers = start_room_positions + start_room_sizes // 2  # (NUM_ROOMS, 2)

    # Pick two forager rooms per team, plus an optional third lone room for
    # non-foragers. All chosen rooms are kept far from one another and from
    # previously assigned team rooms.
    num_teams = static_params.num_teams

    # Pairwise room distances (computed once, used inside the scan closure)
    all_dists = jnp.sqrt(
        ((room_centers[:, None, :] - room_centers[None, :, :]).astype(jnp.float32) ** 2).sum(axis=-1)
    )  # (NUM_ROOMS, NUM_ROOMS)

    def _pick_spawn_room(rng_choice, used_mask):
        min_dist_to_used = jnp.where(
            used_mask[None, :],
            all_dists,
            jnp.float32(1e6),
        ).min(axis=1)
        far_enough = min_dist_to_used >= params.min_team_spawn_distance
        available = jnp.logical_and(far_enough, jnp.logical_not(used_mask))
        unused = jnp.logical_not(used_mask)
        has_valid = available.astype(jnp.float32).sum() > 0
        probs = jnp.where(
            has_valid,
            available.astype(jnp.float32),
            unused.astype(jnp.float32),
        )
        probs = probs / jnp.maximum(probs.sum(), 1.0)
        room_idx = jax.random.choice(rng_choice, NUM_ROOMS, p=probs)
        return used_mask.at[room_idx].set(True), room_idx

    rng, _rng_ra, _rng_rb, _rng_rc, _rng_non_forager = jax.random.split(rng, 5)
    team_rngs_a = jax.random.split(_rng_ra, num_teams)   # for picking forager-room A per team
    team_rngs_b = jax.random.split(_rng_rb, num_teams)   # for picking forager-room B per team
    team_rngs_c = jax.random.split(_rng_rc, num_teams)   # optional lone non-forager room per team
    non_forager_room_slots = jax.random.randint(
        _rng_non_forager,
        shape=(num_teams,),
        minval=0,
        maxval=3 if has_non_forager_lone_room else 2,
        dtype=jnp.int32,
    )

    def _pick_team_rooms(carry, team_idx):
        used_mask = carry  # (NUM_ROOMS,) bool: rooms already taken
        rng_a = team_rngs_a[team_idx]
        rng_b = team_rngs_b[team_idx]
        rng_c = team_rngs_c[team_idx]

        used_with_a, room_idx_a = _pick_spawn_room(rng_a, used_mask)
        used_with_ab, room_idx_b = _pick_spawn_room(rng_b, used_with_a)

        if has_non_forager_lone_room:
            used_with_abc, room_idx_c = _pick_spawn_room(rng_c, used_with_ab)
            return used_with_abc, jnp.stack([room_idx_a, room_idx_b, room_idx_c])

        return used_with_ab, jnp.stack([room_idx_a, room_idx_b, room_idx_b])

    init_used = jnp.zeros(NUM_ROOMS, dtype=bool)
    _, team_room_groups = jax.lax.scan(_pick_team_rooms, init_used, jnp.arange(num_teams))
    # team_room_groups: (num_teams, 3) — [room_A_idx, room_B_idx, room_C_idx]
    # room C is the lone non-forager room when enabled; otherwise it mirrors room B.

    # --- Assign each player to a room ---
    # First forager in TEAM_COMPOSITION  → room A (pair index 0)
    # Second forager in TEAM_COMPOSITION → room B (pair index 1)
    # Non-forager (warrior/miner) → room A, B or lone room C with equal probability.
    spawn_offsets = jnp.array(
        [
            [0, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [-1, 1],
            [1, -1],
            [-1, -1],
            [2, 0],
            [-2, 0],
            [0, 2],
            [0, -2],
        ],
        dtype=jnp.int32,
    )
    player_indices     = jnp.arange(static_params.player_count, dtype=jnp.int32)
    within_team_slot   = player_indices % agents_per_team  # (player_count,)

    is_forager = player_specializations == Specialization.FORAGER.value  # (player_count,)
    # Room-pair slot: first forager in the team composition → 0,
    # second forager → 1, non-forager → one of {0, 1, 2}
    non_forager_slots_per_player = non_forager_room_slots[player_sc]   # (player_count,)
    # Cap to [0,1] so compositions with >2 foragers still map into the two-room layout.
    forager_room_slot = forager_rank_by_slot[within_team_slot]
    room_pair_slot = jnp.where(is_forager, forager_room_slot, non_forager_slots_per_player)  # (player_count,)

    player_room_idx      = team_room_groups[player_sc, room_pair_slot]         # (player_count,)
    player_room_center   = room_centers[player_room_idx]                        # (player_count, 2)
    player_room_pos      = start_room_positions[player_room_idx]                # (player_count, 2)
    player_room_sz       = start_room_sizes[player_room_idx]                    # (player_count, 2)

    # Per-player bounds of the player's own assigned start room, used for auto-respawn checks.
    player_spawn_room_min = player_room_pos
    player_spawn_room_max = player_room_pos + player_room_sz - 1

    slot_indices = within_team_slot % spawn_offsets.shape[0]
    raw_player_position  = player_room_center + spawn_offsets[slot_indices]
    room_min = player_room_pos
    room_max = player_room_pos + player_room_sz - 1
    player_position = jnp.clip(raw_player_position, room_min, room_max)

    # Force spawn tiles to PATH on the start level (clear the spots)
    # Only overwrite if the current block is solid/would trap the player (wall, chest, etc.)
    # Keep fountains and other non-blocking features intact.
    spawn_blocks = map[START_LEVEL, player_position[:, 0], player_position[:, 1]]
    is_solid_spawn = jnp.isin(spawn_blocks, jnp.array(SOLID_BLOCKS))
    map = map.at[START_LEVEL, player_position[:, 0], player_position[:, 1]].set(
        jnp.where(is_solid_spawn, BlockType.PATH.value, spawn_blocks)
    )
    # Only remove ladders from spawn tiles to prevent immediate floor transitions
    spawn_items = item_map[START_LEVEL, player_position[:, 0], player_position[:, 1]]
    is_ladder = jnp.isin(spawn_items, jnp.array([
        ItemType.LADDER_DOWN.value,
        ItemType.LADDER_UP.value,
        ItemType.LADDER_DOWN_BLOCKED.value,
    ]))
    item_map = item_map.at[START_LEVEL, player_position[:, 0], player_position[:, 1]].set(
        jnp.where(is_ladder, ItemType.NONE.value, spawn_items)
    )

    room_slot_count = 3 if has_non_forager_lone_room else 2
    extra_snail_room_active = (
        non_forager_room_slots == 2
        if has_non_forager_lone_room
        else jnp.zeros((num_teams,), dtype=bool)
    )
    active_spawn_room_mask = jnp.concatenate(
        [
            jnp.ones((num_teams, 2), dtype=bool),
            extra_snail_room_active[:, None],
        ],
        axis=1,
    ) if has_non_forager_lone_room else jnp.ones((num_teams, 2), dtype=bool)

    # Remove SNAIL_SPAWN tiles from all active start rooms. This includes the
    # lone non-forager room only when a non-forager actually spawns there.
    active_spawn_room_indices = team_room_groups[:, :room_slot_count].reshape(-1)
    flat_active_spawn_room_mask = active_spawn_room_mask.reshape(-1)
    def _clear_spawn_room_snails(current_map, room_info):
        room_idx, room_active = room_info
        rp = start_room_positions[room_idx]
        rs = start_room_sizes[room_idx]
        rows = jnp.arange(static_params.map_size[0])
        cols = jnp.arange(static_params.map_size[1])
        in_room = (rows >= rp[0])[:, None] & (rows < rp[0] + rs[0])[:, None] & \
                  (cols >= rp[1])[None, :] & (cols < rp[1] + rs[1])[None, :]
        is_snail = current_map[START_LEVEL] == BlockType.SNAIL_SPAWN.value
        revert = in_room & is_snail & room_active
        new_level_map = jnp.where(revert, BlockType.PATH.value, current_map[START_LEVEL])
        current_map = current_map.at[START_LEVEL].set(new_level_map)
        return current_map, None

    map, _ = jax.lax.scan(
        _clear_spawn_room_snails,
        map,
        (active_spawn_room_indices, flat_active_spawn_room_mask),
    )

    # Mobs
    def generate_empty_mobs(max_mobs):
        return Mobs(
            position=jnp.zeros(
                (static_params.num_levels, max_mobs, 2), dtype=jnp.int32
            ),
            health=jnp.ones((static_params.num_levels, max_mobs), dtype=jnp.float32),
            mask=jnp.zeros((static_params.num_levels, max_mobs), dtype=bool),
            attack_cooldown=jnp.zeros(
                (static_params.num_levels, max_mobs), dtype=jnp.int32
            ),
            type_id=jnp.zeros((static_params.num_levels, max_mobs), dtype=jnp.int32),
        )

    melee_mobs = generate_empty_mobs(
        static_params.max_melee_mobs
    )
    ranged_mobs = generate_empty_mobs(
        static_params.max_ranged_mobs
    )
    passive_mobs = generate_empty_mobs(
        static_params.max_passive_mobs
    )

    # Pre-spawn 1-3 snails per team in each team's spawn room
    MAX_SNAILS_PER_TEAM = 3
    snail_type_id = FLOOR_MOB_MAPPING[START_LEVEL, MobType.PASSIVE.value]
    snail_health = MOB_TYPE_HEALTH_MAPPING[snail_type_id, MobType.PASSIVE.value]
    snail_spawn_offsets = jnp.array(
        [
            [0, 1],
            [0, -1],
            [1, 1],
            [-1, 1],
            [1, -1],
            [-1, -1],
            [2, 1],
            [-2, 1],
            [2, -1],
            [-2, -1],
            [2, 0],
            [-2, 0],
            [0, 2],
            [0, -2],
        ],
        dtype=jnp.int32,
    )
    # Spawn 1-3 snails per active start room. The third room is only populated
    # when a non-forager actually spawns there alone.
    rng, _snail_rng = jax.random.split(rng)
    snail_count_rngs = jax.random.split(_snail_rng, num_teams * room_slot_count)
    for t in range(num_teams):
        for f in range(room_slot_count):
            room_idx_tf = team_room_groups[t, f]
            forager_center = room_centers[room_idx_tf]
            room_min = start_room_positions[room_idx_tf]
            room_max = room_min + start_room_sizes[room_idx_tf] - 1
            room_is_active = active_spawn_room_mask[t, f]

            # Randomly choose 1-3 snails for this active room.
            num_snails = jax.random.randint(
                snail_count_rngs[t * room_slot_count + f],
                (),
                1,
                MAX_SNAILS_PER_TEAM + 1,
            )

            for s in range(MAX_SNAILS_PER_TEAM):
                should_spawn = jnp.logical_and(s < num_snails, room_is_active)
                default_pos = jnp.clip(forager_center + snail_spawn_offsets[0], room_min, room_max)
                snail_pos = default_pos
                has_selected_pos = jnp.asarray(False)

                # Pick a small offset near the room center not occupied by a player or another snail.
                for offset in snail_spawn_offsets:
                    candidate_pos = jnp.clip(forager_center + offset, room_min, room_max)
                    collides_with_player = (player_position == candidate_pos[None, :]).all(axis=1).any()
                    # Check collision with already-placed snails for this forager room
                    collides_with_snail = jnp.asarray(False)
                    for prev_s in range(s):
                        prev_mob_idx = (t * room_slot_count + f) * MAX_SNAILS_PER_TEAM + prev_s
                        prev_pos = passive_mobs.position[START_LEVEL, prev_mob_idx]
                        prev_active = passive_mobs.mask[START_LEVEL, prev_mob_idx]
                        same_pos = jnp.logical_and(prev_active, (candidate_pos == prev_pos).all())
                        collides_with_snail = jnp.logical_or(collides_with_snail, same_pos)
                    candidate_block = map[START_LEVEL, candidate_pos[0], candidate_pos[1]]
                    is_walkable = jnp.logical_not(jnp.isin(candidate_block, jnp.array(SOLID_BLOCKS)))
                    no_collision = jnp.logical_and(
                        jnp.logical_not(collides_with_player),
                        jnp.logical_not(collides_with_snail),
                    )
                    can_use_candidate = jnp.logical_and(no_collision, is_walkable)
                    take_candidate = jnp.logical_and(jnp.logical_not(has_selected_pos), can_use_candidate)
                    snail_pos = jnp.where(take_candidate, candidate_pos, snail_pos)
                    has_selected_pos = jnp.logical_or(has_selected_pos, take_candidate)

                mob_idx = (t * room_slot_count + f) * MAX_SNAILS_PER_TEAM + s
                passive_mobs = passive_mobs.replace(
                    position=passive_mobs.position.at[START_LEVEL, mob_idx].set(
                        jnp.where(should_spawn, snail_pos, passive_mobs.position[START_LEVEL, mob_idx])),
                    health=passive_mobs.health.at[START_LEVEL, mob_idx].set(
                        jnp.where(should_spawn, snail_health, passive_mobs.health[START_LEVEL, mob_idx])),
                    mask=passive_mobs.mask.at[START_LEVEL, mob_idx].set(
                        jnp.where(should_spawn, True, passive_mobs.mask[START_LEVEL, mob_idx])),
                    type_id=passive_mobs.type_id.at[START_LEVEL, mob_idx].set(
                        jnp.where(should_spawn, snail_type_id, passive_mobs.type_id[START_LEVEL, mob_idx])),
                )

    warrior_spawn_room_mask = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.logical_and(
                        jnp.logical_and(player_sc == t, room_pair_slot == s),
                        player_specializations == Specialization.WARRIOR.value,
                    ).any()
                    for s in range(room_slot_count)
                ]
            )
            for t in range(num_teams)
        ]
    )

    # Per team, spawn 0-1 melee predator with 50% probability. By default the
    # predator is sampled from all active start rooms; experiments can restrict
    # this t=0 seeding to the room where the warrior actually starts.
    melee_type_id = FLOOR_MOB_MAPPING[START_LEVEL, MobType.MELEE.value]
    melee_health = MOB_TYPE_HEALTH_MAPPING[melee_type_id, MobType.MELEE.value]
    melee_spawn_offsets = jnp.array(
        [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0],
            [1, 1],
            [-1, 1],
            [1, -1],
            [-1, -1],
            [2, 0],
            [-2, 0],
            [0, 2],
            [0, -2],
        ],
        dtype=jnp.int32,
    )
    rng, _melee_rng = jax.random.split(rng)
    melee_team_rngs = jax.random.split(_melee_rng, num_teams * 2)
    for t in range(num_teams):
        should_spawn_melee = jax.random.bernoulli(melee_team_rngs[t * 2], 0.5)
        active_room_slot_probs = active_spawn_room_mask[t].astype(jnp.float32)
        warrior_room_slot_probs = jnp.logical_and(
            active_spawn_room_mask[t],
            warrior_spawn_room_mask[t],
        ).astype(jnp.float32)
        has_warrior_room = warrior_room_slot_probs.sum() > 0
        restrict_to_warrior_room = jnp.logical_and(
            params.initial_predators_spawn_in_warrior_rooms_only,
            has_warrior_room,
        )
        room_slot_probs = jnp.where(
            restrict_to_warrior_room,
            warrior_room_slot_probs,
            active_room_slot_probs,
        )
        room_slot_probs = room_slot_probs / jnp.maximum(room_slot_probs.sum(), 1.0)
        should_spawn_melee = jnp.logical_and(
            should_spawn_melee,
            jnp.logical_or(
                jnp.logical_not(params.initial_predators_spawn_in_warrior_rooms_only),
                has_warrior_room,
            ),
        )
        chosen_room_slot = jax.random.choice(
            melee_team_rngs[t * 2 + 1],
            jnp.arange(room_slot_count),
            p=room_slot_probs,
        )
        chosen_room_idx = team_room_groups[t, chosen_room_slot]
        chosen_room_center = room_centers[chosen_room_idx]
        room_min = start_room_positions[chosen_room_idx]
        room_max = room_min + start_room_sizes[chosen_room_idx] - 1

        default_pos = jnp.clip(chosen_room_center + melee_spawn_offsets[0], room_min, room_max)
        melee_pos = default_pos
        has_selected_pos = jnp.asarray(False)

        for offset in melee_spawn_offsets:
            candidate_pos = jnp.clip(chosen_room_center + offset, room_min, room_max)
            collides_with_player = (player_position == candidate_pos[None, :]).all(axis=1).any()
            collides_with_snail = jnp.logical_and(
                passive_mobs.mask[START_LEVEL],
                (passive_mobs.position[START_LEVEL] == candidate_pos[None, :]).all(axis=1),
            ).any()
            collides_with_melee = jnp.logical_and(
                melee_mobs.mask[START_LEVEL],
                (melee_mobs.position[START_LEVEL] == candidate_pos[None, :]).all(axis=1),
            ).any()
            candidate_block = map[START_LEVEL, candidate_pos[0], candidate_pos[1]]
            is_walkable = jnp.logical_not(jnp.isin(candidate_block, jnp.array(SOLID_BLOCKS)))
            no_collision = jnp.logical_and(
                jnp.logical_not(collides_with_player),
                jnp.logical_and(
                    jnp.logical_not(collides_with_snail),
                    jnp.logical_not(collides_with_melee),
                ),
            )
            can_use_candidate = jnp.logical_and(no_collision, is_walkable)
            take_candidate = jnp.logical_and(jnp.logical_not(has_selected_pos), can_use_candidate)
            melee_pos = jnp.where(take_candidate, candidate_pos, melee_pos)
            has_selected_pos = jnp.logical_or(has_selected_pos, take_candidate)

        should_spawn_melee = jnp.logical_and(should_spawn_melee, has_selected_pos)
        melee_mobs = melee_mobs.replace(
            position=melee_mobs.position.at[START_LEVEL, t].set(
                jnp.where(should_spawn_melee, melee_pos, melee_mobs.position[START_LEVEL, t])
            ),
            health=melee_mobs.health.at[START_LEVEL, t].set(
                jnp.where(should_spawn_melee, melee_health, melee_mobs.health[START_LEVEL, t])
            ),
            mask=melee_mobs.mask.at[START_LEVEL, t].set(
                jnp.where(should_spawn_melee, True, melee_mobs.mask[START_LEVEL, t])
            ),
            type_id=melee_mobs.type_id.at[START_LEVEL, t].set(
                jnp.where(should_spawn_melee, melee_type_id, melee_mobs.type_id[START_LEVEL, t])
            ),
        )

    # Projectiles
    def _create_projectiles(max_num):
        projectiles = generate_empty_mobs(max_num)

        projectile_directions = jnp.ones(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )

        projectile_owners = jnp.zeros(
            (static_params.num_levels, max_num), dtype=jnp.int32
        )

        return projectiles, projectile_directions, projectile_owners

    mob_projectiles, mob_projectile_directions, mob_projectile_owners = _create_projectiles(
        static_params.max_mob_projectiles
    )
    player_projectiles, player_projectile_directions, player_projectile_owners = _create_projectiles(
        static_params.max_player_projectiles
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

    # Potion mapping for episode
    rng, _rng = jax.random.split(rng)
    potion_mapping = jax.random.permutation(_rng, jnp.arange(6))

    # Inventory
    inventory = jax.tree_util.tree_map(
        lambda x, y: jax.lax.select(params.god_mode, x, y),
        get_new_full_inventory(static_params.player_count),
        get_new_empty_inventory(static_params.player_count),
    )

    rng, _rng = jax.random.split(rng)

    state = EnvState(
        map=map,
        item_map=item_map,
        mob_map=jnp.zeros(
            (static_params.num_levels, *static_params.map_size), dtype=bool
        ),
        light_map=light_map,
        down_ladders=ladders_down,
        up_ladders=ladders_up,
        chests_opened=jnp.zeros((static_params.num_levels, static_params.player_count), dtype=bool),
        monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
        .at[0]
        .set(10),  # First ladder starts open
        player_position=player_position,
        player_spawn_position=player_position,
        player_direction=jnp.full(
            (static_params.player_count,), Action.UP.value, dtype=jnp.int32
        ),
        player_level=jnp.asarray(2, dtype=jnp.int32),
        player_health=jnp.full((static_params.player_count,), 9.0, dtype=jnp.float32),
        player_alive=jnp.full((static_params.player_count,), True, dtype=bool),
        player_food=jnp.full((static_params.player_count,), 9, dtype=jnp.int32),
        player_drink=jnp.full((static_params.player_count,), 9, dtype=jnp.int32),
        player_energy=jnp.full((static_params.player_count,), 9, dtype=jnp.int32),
        player_mana=jnp.full((static_params.player_count,), 9, dtype=jnp.int32),
        player_recover=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_hunger=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_thirst=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_fatigue=jnp.full((static_params.player_count,), 0.0, dtype=jnp.float32),
        player_recover_mana=jnp.full(
            (static_params.player_count,), 0.0, dtype=jnp.float32
        ),
        is_sleeping=jnp.full((static_params.player_count,), False, dtype=jnp.bool),
        is_resting=jnp.full((static_params.player_count,), False, dtype=jnp.bool),
        player_xp=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        player_dexterity=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        player_strength=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        player_intelligence=jnp.full((static_params.player_count,), 1, dtype=jnp.int32),
        player_specialization=player_specializations,
        player_sc = player_sc,
        request_duration=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        request_type=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        inventory=inventory,
        sword_enchantment=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        bow_enchantment=jnp.full((static_params.player_count,), 0, dtype=jnp.int32),
        armour_enchantments=jnp.full(
            (static_params.player_count, 4), 0, dtype=jnp.int32
        ),
        melee_mobs=melee_mobs,
        ranged_mobs=ranged_mobs,
        passive_mobs=passive_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_projectile_directions,
        mob_projectile_owners=mob_projectile_owners,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_projectile_directions,
        player_projectile_owners=player_projectile_owners,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        potion_mapping=potion_mapping,
        learned_spells=jnp.full((static_params.player_count,), False, dtype=jnp.bool),
        boss_progress=jnp.asarray(0, dtype=jnp.int32),
        boss_timesteps_to_spawn_this_round=jnp.asarray(
            BOSS_FIGHT_SPAWN_TURNS, dtype=jnp.int32
        ),
        achievements=jnp.zeros(
            (static_params.player_count, len(Achievement)), dtype=bool
        ),
        light_level=jnp.asarray(calculate_light_level(0, params), dtype=jnp.float32),
        trade_count=jnp.asarray(0, dtype=jnp.int32),
        food_trade_count=jnp.asarray(0, dtype=jnp.int32),
        drink_trade_count=jnp.asarray(0, dtype=jnp.int32),
        revives=jnp.asarray(0, dtype=jnp.int32),
        revive_cooldown_until=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        trade_give_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        trade_receive_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        revive_as_reviver_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        revive_as_revived_count=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        team_kills=jnp.zeros(static_params.num_teams, dtype=jnp.int32),
        walking_distance=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        sum_distance_to_spawn=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_melee=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health_food=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health_drink=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health_energy=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health_other=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_ff=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        consecutive_dead_steps=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        damage_dealt_to_other_team=jnp.zeros((static_params.num_teams,), dtype=jnp.float32),
        individual_reward_return=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        log_trade_give=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        log_trade_receive=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        log_trade_give_material_id=jnp.full((static_params.player_count,), -1, dtype=jnp.int32),
        log_trade_receive_material_id=jnp.full((static_params.player_count,), -1, dtype=jnp.int32),
        log_trade_give_partner_id=jnp.full((static_params.player_count,), -1, dtype=jnp.int32),
        log_trade_receive_partner_id=jnp.full((static_params.player_count,), -1, dtype=jnp.int32),
        log_revive_as_reviver=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        log_revive_as_revived=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        log_revive_partner_id=jnp.full((static_params.player_count,), -1, dtype=jnp.int32),
        log_melee_kills=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        log_auto_respawned=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        player_spawn_room_min=player_spawn_room_min.astype(jnp.int32),
        player_spawn_room_max=player_spawn_room_max.astype(jnp.int32),
        effective_max_timesteps=jnp.asarray(params.max_timesteps, dtype=jnp.float32),
        state_rng=_rng,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )

    # def print_agent_stats(i, spec, sc):
    #     jax.debug.print("AGENT {i} INITIALIZED: Spec={s}, Subclass={c}", 
    #                     i=i, s=spec, c=sc)
        
    # jax.vmap(print_agent_stats)(
    #     jnp.arange(static_params.player_count), 
    #     player_specializations, 
    #     player_sc
    # )

    return state
