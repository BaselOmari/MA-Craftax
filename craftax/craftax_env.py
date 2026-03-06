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


def _validate_team_config(num_teams: int, team_composition: tuple):
    if num_teams <= 0:
        raise ValueError(f"num_teams must be >= 1, got {num_teams}.")
    if len(team_composition) == 0:
        raise ValueError("team_composition must contain at least one role id.")

def make_craftax_env_from_name(name: str, num_teams: int = 2, team_composition: tuple = (1, 1, 2)):
    if name == "Craftax-Coop-Symbolic":
        _validate_team_config(num_teams, team_composition)
        return CraftaxCoopSymbolicEnv(num_teams=num_teams, team_composition=team_composition)
    elif name == "Craftax-Coop-Pixels":
        _validate_team_config(num_teams, team_composition)
        return CraftaxCoopPixelsEnv(num_teams=num_teams, team_composition=team_composition)
    elif name == "Craftax-MA-Symbolic":
        return CraftaxMASymbolicEnv()
    elif name == "Craftax-MA-Pixels":
        return CraftaxMAPixelsEnv()

    raise ValueError(f"Unknown multi-agent craftax environment: {name}")
