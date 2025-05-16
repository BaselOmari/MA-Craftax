from craftax.craftax_coop.envs.craftax_symbolic_env import (
    CraftaxCoopSymbolicEnv,
)
from craftax.craftax_coop.envs.craftax_pixels_env import (
    CraftaxCoopPixelsEnv,
)
from craftax.craftax_ma.envs.craftax_symbolic_env import (
    CraftaxMASymbolicEnv,
)
from craftax.craftax_ma.envs.craftax_pixels_env import (
    CraftaxMAPixelsEnv,
)

def make_craftax_env_from_name(name: str):
    if name == "Craftax-Coop-Symbolic":
        return CraftaxCoopSymbolicEnv()
    elif name == "Craftax-Coop-Pixels":
        return CraftaxCoopPixelsEnv()
    elif name == "Craftax-MA-Symbolic":
        return CraftaxMASymbolicEnv()
    elif name == "Craftax-MA-Pixels":
        return CraftaxMAPixelsEnv()

    raise ValueError(f"Unknown multi-agent craftax environment: {name}")
