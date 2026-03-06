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

    def _add_path(carry, path_index):
        cmap, included_rooms_mask, rng = carry

        path_source = room_positions[path_index]

        rng, _rng = jax.random.split(rng)
        sink_index = jax.random.choice(
            _rng, jnp.arange(NUM_ROOMS), p=included_rooms_mask
        )
        path_sink = room_positions[sink_index]

        # Horizontal component
        entire_row = cmap[path_source[0]]
        path_indexes = jnp.arange(static_params.map_size[0] + 2 * MAX_ROOM_SIZE)
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
            horizontal_mask, entire_row == BlockType.WALL.value
        )

        new_row = (
            horizontal_mask * BlockType.PATH.value + (1 - horizontal_mask) * entire_row
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_row, axis=0),
            path_source,
        )

        # Vertical component
        entire_col = cmap[:, path_sink[1]]
        path_indexes = jnp.arange(static_params.map_size[1] + 2 * MAX_ROOM_SIZE)
        path_indexes = path_indexes - path_source[0]
        vertical_distance = path_sink[0] - path_source[0]
        path_indexes = path_indexes * jnp.sign(vertical_distance)

        vertical_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(vertical_distance)
        )
        vertical_mask = jnp.logical_and(vertical_mask, jnp.sign(vertical_distance))

        vertical_mask = jnp.logical_and(
            vertical_mask, entire_col == BlockType.WALL.value
        )

        new_col = (
            vertical_mask * BlockType.PATH.value + (1 - vertical_mask) * entire_col
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_col, axis=-1),
            path_sink,
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
    if static_params.num_teams > NUM_ROOMS:
        raise ValueError(
            f"num_teams ({static_params.num_teams}) exceeds available rooms "
            f"({NUM_ROOMS}) for team spawn assignment."
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

    # Fix player specializations from team_composition config
    # e.g. team_composition=(1, 1, 2) with num_teams=2 -> [1, 1, 2, 1, 1, 2]
    comp = jnp.array(static_params.team_composition)
    player_specializations = jnp.tile(comp, static_params.num_teams)

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

    # Pick one room per team, each far from previously chosen rooms
    num_teams = static_params.num_teams
    team_rngs = jax.random.split(rng, num_teams + 1)
    rng = team_rngs[0]
    team_rngs = team_rngs[1:]

    def _pick_team_room(carry, team_idx):
        used_mask = carry  # (NUM_ROOMS,) bool: rooms already taken
        team_rng = team_rngs[team_idx]

        # Compute min distance to all already-chosen rooms
        # For the first team, no rooms are used yet so all are valid
        all_dists = jnp.sqrt(
            ((room_centers[:, None, :] - room_centers[None, :, :]).astype(jnp.float32) ** 2).sum(axis=-1)
        )  # (NUM_ROOMS, NUM_ROOMS)
        # Min distance from each candidate to any used room (large value if no rooms used yet)
        min_dist_to_used = jnp.where(used_mask[None, :], all_dists, jnp.float32(1e6)).min(axis=1)  # (NUM_ROOMS,)
        # First team: no rooms used, so min_dist_to_used is all 1e6 -> all far enough
        far_enough = min_dist_to_used >= params.min_team_spawn_distance
        # Exclude already-used rooms
        available = jnp.logical_and(far_enough, jnp.logical_not(used_mask))

        # Fallback: if no room is far enough, use any unused room
        unused = jnp.logical_not(used_mask)
        has_valid = available.astype(jnp.float32).sum() > 0
        probs = jnp.where(has_valid, available.astype(jnp.float32), unused.astype(jnp.float32))
        probs = probs / jnp.maximum(probs.sum(), 1.0)

        room_idx = jax.random.choice(team_rng, NUM_ROOMS, p=probs)
        new_used_mask = used_mask.at[room_idx].set(True)
        return new_used_mask, room_idx

    init_used = jnp.zeros(NUM_ROOMS, dtype=bool)
    _, team_room_indices = jax.lax.scan(_pick_team_room, init_used, jnp.arange(num_teams))
    # team_room_indices: (num_teams,) - room index for each team

    team_centers_all = room_centers[team_room_indices]  # (num_teams, 2)
    team_room_positions_all = start_room_positions[team_room_indices]  # (num_teams, 2)
    team_room_sizes_all = start_room_sizes[team_room_indices]  # (num_teams, 2)

    # Assign each player to a nearby tile in its team's room (avoid full stacking)
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
    player_indices = jnp.arange(static_params.player_count, dtype=jnp.int32)
    # Slot within team (0, 1, 2, ..., agents_per_team-1, 0, 1, 2, ...)
    slot_indices = (player_indices % agents_per_team) % spawn_offsets.shape[0]

    # Map each player to its team's room center and bounds
    team_centers = team_centers_all[player_sc]  # (player_count, 2)
    raw_player_position = team_centers + spawn_offsets[slot_indices]

    team_room_positions = team_room_positions_all[player_sc]  # (player_count, 2)
    team_room_sizes = team_room_sizes_all[player_sc]  # (player_count, 2)
    room_min = team_room_positions
    room_max = team_room_positions + team_room_sizes - 1
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

    # Remove SNAIL_SPAWN tiles from team spawn rooms
    # team_room_indices: (num_teams,) indices into dungeon_room_positions[0]
    def _clear_spawn_room_snails(current_map, team_idx):
        rp = start_room_positions[team_room_indices[team_idx]]
        rs = start_room_sizes[team_room_indices[team_idx]]
        rows = jnp.arange(static_params.map_size[0])
        cols = jnp.arange(static_params.map_size[1])
        in_room = (rows >= rp[0])[:, None] & (rows < rp[0] + rs[0])[:, None] & \
                  (cols >= rp[1])[None, :] & (cols < rp[1] + rs[1])[None, :]
        is_snail = current_map[START_LEVEL] == BlockType.SNAIL_SPAWN.value
        revert = in_room & is_snail
        new_level_map = jnp.where(revert, BlockType.PATH.value, current_map[START_LEVEL])
        current_map = current_map.at[START_LEVEL].set(new_level_map)
        return current_map, None

    map, _ = jax.lax.scan(_clear_spawn_room_snails, map, jnp.arange(num_teams))

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
        static_params.max_melee_mobs * static_params.player_count
    )
    ranged_mobs = generate_empty_mobs(
        static_params.max_ranged_mobs * static_params.player_count
    )
    passive_mobs = generate_empty_mobs(
        static_params.max_passive_mobs * static_params.player_count
    )

    # Pre-spawn one snail per team in each team's spawn room
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
    for t in range(num_teams):
        room_min = team_room_positions_all[t]
        room_max = team_room_positions_all[t] + team_room_sizes_all[t] - 1
        default_pos = jnp.clip(team_centers_all[t] + jnp.array([0, 1]), room_min, room_max)
        snail_pos = default_pos
        has_selected_pos = jnp.asarray(False)

        # Pick a small offset near the room center that is not occupied by a player.
        for offset in snail_spawn_offsets:
            candidate_pos = jnp.clip(team_centers_all[t] + offset, room_min, room_max)
            collides_with_player = (player_position == candidate_pos[None, :]).all(axis=1).any()
            candidate_block = map[START_LEVEL, candidate_pos[0], candidate_pos[1]]
            is_walkable = jnp.logical_not(jnp.isin(candidate_block, jnp.array(SOLID_BLOCKS)))
            can_use_candidate = jnp.logical_and(jnp.logical_not(collides_with_player), is_walkable)
            take_candidate = jnp.logical_and(jnp.logical_not(has_selected_pos), can_use_candidate)
            snail_pos = jnp.where(take_candidate, candidate_pos, snail_pos)
            has_selected_pos = jnp.logical_or(has_selected_pos, take_candidate)

        passive_mobs = passive_mobs.replace(
            position=passive_mobs.position.at[START_LEVEL, t].set(snail_pos),
            health=passive_mobs.health.at[START_LEVEL, t].set(snail_health),
            mask=passive_mobs.mask.at[START_LEVEL, t].set(True),
            type_id=passive_mobs.type_id.at[START_LEVEL, t].set(snail_type_id),
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
        static_params.max_mob_projectiles * static_params.player_count
    )
    player_projectiles, player_projectile_directions, player_projectile_owners = _create_projectiles(
        static_params.max_player_projectiles * static_params.player_count
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants * static_params.player_count, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants * static_params.player_count, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants * static_params.player_count, dtype=bool)

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
        wood_trade_count=jnp.asarray(0, dtype=jnp.int32),
        same_trade_count=jnp.asarray(0, dtype=jnp.int32),
        revives=jnp.asarray(0, dtype=jnp.int32),
        ff_damage_dealt=jnp.asarray(0.0, dtype=jnp.float32),
        team_kills=jnp.zeros(static_params.num_teams, dtype=jnp.int32),
        walking_distance=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        ticks_moved=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        ticks_tried_moving=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        damage_taken_total=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_melee=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_ranged=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health_food=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health_drink=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health_energy=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_health_other=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        damage_taken_ff=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        ticks_food_empty=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        ticks_drink_empty=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        ticks_energy_empty=jnp.zeros((static_params.player_count,), dtype=jnp.int32),
        steps_alive=jnp.zeros((static_params.player_count,), dtype=jnp.float32),
        team_alive_time=jnp.zeros((static_params.num_teams,), dtype=jnp.float32),
        damage_dealt_to_other_team=jnp.zeros((static_params.num_teams,), dtype=jnp.float32),
        all_necessities_frac=jnp.ones((static_params.player_count,), dtype=jnp.float32),
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
