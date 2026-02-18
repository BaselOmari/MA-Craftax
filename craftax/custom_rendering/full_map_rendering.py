import jax
import jax.numpy as jnp


def render_full_map(state, static_params, textures, player_specific_textures, block_pixel_size=16, env_name="Craftax-Coop-Symbolic"):
    if "Coop" in env_name:
        from craftax.craftax_coop.constants import (
            BlockType, ItemType, MONSTERS_KILLED_TO_CLEAR_LEVEL
        )
        from craftax.craftax_coop.util.game_logic_utils import is_boss_vulnerable
    else:
        from craftax.craftax_ma.constants import (
            BlockType, ItemType, MONSTERS_KILLED_TO_CLEAR_LEVEL
        )
        from craftax.craftax_ma.util.game_logic_utils import is_boss_vulnerable
    
    # Get the full map for current floor (no viewport slicing)
    map_full = state.map[state.player_level]  # Shape: (48, 48)
    map_size = map_full.shape
    
    # Boss block handling
    boss_block = jax.lax.select(
        is_boss_vulnerable(state),
        BlockType.NECROMANCER_VULNERABLE.value,
        BlockType.NECROMANCER.value,
    )
    map_view_boss = map_full == BlockType.NECROMANCER.value
    map_full = map_view_boss * boss_block + (1 - map_view_boss) * map_full
    
    # Initialize pixel array
    map_pixels = jnp.zeros(
        (map_size[0] * block_pixel_size, map_size[1] * block_pixel_size, 3),
        dtype=jnp.float32
    )
    
    # Render each block type by tiling individual block textures
    def _add_block_type_to_pixels(pixels, block_index):
        # Get positions where this block type appears
        block_mask = map_full == block_index  # Shape: (48, 48)
        
        # Get the texture for this block type
        block_texture = textures["block_textures"][block_index]  # Shape: (16, 16, 3)
        
        # Create full-size texture by repeating for each block position
        # We'll scan through each tile position
        def render_tile(y_tile, x_tile):
            # Calculate pixel position for this tile
            y_start = y_tile * block_pixel_size
            x_start = x_tile * block_pixel_size
            
            # Return texture if this position has the block, else zeros
            return jax.lax.select(
                block_mask[y_tile, x_tile],
                block_texture,
                jnp.zeros_like(block_texture)
            )
        
        # Vectorize over all tile positions
        y_indices = jnp.arange(map_size[0])
        x_indices = jnp.arange(map_size[1])
        
        # Use vmap to efficiently render all tiles
        render_row = jax.vmap(render_tile, in_axes=(None, 0))
        render_all = jax.vmap(render_row, in_axes=(0, None))
        
        tiled_textures = render_all(y_indices, x_indices)  # Shape: (48, 48, 16, 16, 3)
        
        # Reshape to full map: (48, 48, 16, 16, 3) -> (48*16, 48*16, 3)
        full_texture = tiled_textures.transpose(0, 2, 1, 3, 4).reshape(
            map_size[0] * block_pixel_size,
            map_size[1] * block_pixel_size,
            3
        )
        
        return pixels + full_texture, None
    
    map_pixels, _ = jax.lax.scan(
        _add_block_type_to_pixels, map_pixels, jnp.arange(len(BlockType))
    )
    
    # Add items
    item_map_full = state.item_map[state.player_level]
    
    # Insert blocked/open ladders
    is_ladder_down_open = (
        state.monsters_killed[state.player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL
    )
    ladder_down_item = jax.lax.select(
        is_ladder_down_open,
        ItemType.LADDER_DOWN.value,
        ItemType.LADDER_DOWN_BLOCKED.value,
    )
    item_map_view_is_ladder_down = item_map_full == ItemType.LADDER_DOWN.value
    item_map_full = (
        item_map_view_is_ladder_down * ladder_down_item
        + (1 - item_map_view_is_ladder_down) * item_map_full
    )
    
    # Add items with tiled textures
    def _add_item_type_to_pixels(pixels, item_index):
        # Get positions where this item type appears
        item_mask = item_map_full == item_index
        
        # Extract single item texture from full_map_item_textures (which is tiled for ego-view)
        # full_map_item_textures has shape (num_items, 9*16, 11*16, 4) for block_pixel_size=16
        # We need just one tile (16x16x4)
        full_item_tex = textures["full_map_item_textures"][item_index]
        item_texture = full_item_tex[:block_pixel_size, :block_pixel_size, :]  # Extract first tile
        
        # Render all tiles with this item
        def render_item_tile(y_tile, x_tile):
            return jax.lax.select(
                item_mask[y_tile, x_tile],
                item_texture,
                jnp.zeros_like(item_texture)
            )
        
        y_indices = jnp.arange(map_size[0])
        x_indices = jnp.arange(map_size[1])
        
        render_row = jax.vmap(render_item_tile, in_axes=(None, 0))
        render_all = jax.vmap(render_row, in_axes=(0, None))
        
        tiled_items = render_all(y_indices, x_indices)  # Shape: (48, 48, 16, 16, 4)
        
        # Reshape to full map
        full_item_texture = tiled_items.transpose(0, 2, 1, 3, 4).reshape(
            map_size[0] * block_pixel_size,
            map_size[1] * block_pixel_size,
            4
        )
        
        # Alpha blend: remove background and add item
        alpha = full_item_texture[:, :, 3:4]
        rgb = full_item_texture[:, :, :3]
        pixels = pixels * (1 - alpha) + rgb * alpha
        
        return pixels, None
    
    map_pixels, _ = jax.lax.scan(
        _add_item_type_to_pixels, map_pixels, jnp.arange(1, len(ItemType))
    )
    
    # Render all players on the map
    def _slice_pixel_map(player_pixels, position):
        return jax.lax.dynamic_slice(
            player_pixels,
            (
                position[0] * block_pixel_size,
                position[1] * block_pixel_size,
                0,
            ),
            (block_pixel_size, block_pixel_size, 3),
        )
    
    def _update_slice_pixel_map(player_pixels, texture_with_background, position):
        return jax.lax.dynamic_update_slice(
            player_pixels,
            texture_with_background,
            (
                position[0] * block_pixel_size,
                position[1] * block_pixel_size,
                0,
            ),
        )
    
    def _render_player(pixels, player_index):
        position = state.player_position[player_index]
        
        # Get player texture based on state
        player_texture_index = jax.lax.select(
            state.is_sleeping[player_index], 4, state.player_direction[player_index] - 1
        )
        player_texture_index = jax.lax.select(
            state.player_alive[player_index], player_texture_index, 5
        )
        player_texture = player_specific_textures.player_textures[player_index, player_texture_index]
        player_texture, player_texture_alpha = (
            player_texture[:, :, :3],
            player_texture[:, :, 3:],
        )
        
        # Blend with background
        background = _slice_pixel_map(pixels, position)
        player_texture_with_background = (1 - player_texture_alpha) * background
        player_texture_with_background = (
            player_texture_with_background + player_texture * player_texture_alpha
        )
        
        pixels = _update_slice_pixel_map(pixels, player_texture_with_background, position)
        return pixels, None
    
    map_pixels, _ = jax.lax.scan(
        _render_player, map_pixels, jnp.arange(static_params.player_count)
    )
    
    # Render mobs (melee, ranged, passive)
    def _add_mob_to_pixels(carry, mob_index):
        pixels, mobs, texture_name, alpha_texture_name = carry
        position = mobs.position[state.player_level, mob_index]
        is_on_map = mobs.mask[state.player_level, mob_index]
        
        mob_texture = texture_name[mobs.type_id[state.player_level, mob_index]]
        mob_texture_alpha = alpha_texture_name[mobs.type_id[state.player_level, mob_index]]
        
        # Only render if mob is active
        background = _slice_pixel_map(pixels, position)
        mob_texture_with_background = (1 - mob_texture_alpha * is_on_map) * background
        mob_texture_with_background = (
            mob_texture_with_background + mob_texture * mob_texture_alpha * is_on_map
        )
        
        pixels = jax.lax.cond(
            is_on_map,
            lambda p: _update_slice_pixel_map(p, mob_texture_with_background, position),
            lambda p: p,
            pixels
        )
        
        return (pixels, mobs, texture_name, alpha_texture_name), None
    
    # Melee mobs
    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (map_pixels, state.melee_mobs, textures["melee_mob_textures"], textures["melee_mob_texture_alphas"]),
        jnp.arange(state.melee_mobs.mask.shape[1]),
    )
    
    # Passive mobs
    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (map_pixels, state.passive_mobs, textures["passive_mob_textures"], textures["passive_mob_texture_alphas"]),
        jnp.arange(state.passive_mobs.mask.shape[1]),
    )
    
    # Ranged mobs
    (map_pixels, _, _, _), _ = jax.lax.scan(
        _add_mob_to_pixels,
        (map_pixels, state.ranged_mobs, textures["ranged_mob_textures"], textures["ranged_mob_texture_alphas"]),
        jnp.arange(state.ranged_mobs.mask.shape[1]),
    )
    
    # Apply lighting (underground darkness)
    light_map = state.light_map[state.player_level]
    light_map_pixels = light_map.repeat(block_pixel_size, axis=0).repeat(
        block_pixel_size, axis=1
    )
    map_pixels = light_map_pixels[:, :, None] * map_pixels
    
    # Apply day/night cycle (only on floor 0 - surface)
    daylight = state.light_level
    daylight = jax.lax.select(state.player_level == 0, daylight, 1.0)
    
    # Simple night darkening without noise for full map view
    night_intensity = 2 * (0.5 - daylight)
    night_intensity = jnp.maximum(night_intensity, 0.0)
    night_darkening = 1.0 - (night_intensity * 0.5)
    map_pixels = night_darkening * map_pixels
    
    return map_pixels
