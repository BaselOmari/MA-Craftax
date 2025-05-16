from typing import List

from craftax_ma.craftax_state import EnvState, StaticEnvParams
from craftax_ma.constants import *

def compute_score(state: EnvState, done: bool, static_params: StaticEnvParams):
    achievements = state.achievements * done * 100.0
    info = {}
    for achievement in Achievement:
        achievement_name = f"Achievements/{achievement.name.lower()}"
        info[achievement_name] = achievements[:, achievement.value]
    return info
