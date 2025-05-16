# Multi-Agent Craftax
We introduce Craftax-MA and Craftax-Coop, MARL environments written entirely in JAX. Craftax-MA reimplements the exact game mechanics as Craftax, while Craftax-Coop introduces agent heterogeniety, trading and other mechanics that require cooperation for success!

## Basic Usage
Craftax-MA and Craftax-Coop conform to the JaxMARL interface, and can be simply used as follows
```python
import jax
from craftax.craftax_env import make_craftax_env_from_name

rng = jax.random.PRNGKey(0)
rng_reset, rng_act, rng_step = jax.random.split(rng, 3)

# Create environment
env = make_craftax_env_from_name("Craftax-Coop-Symbolic")

# Get an initial state and observation
obs, states = env.reset(rng_reset)

# Pick random actions
rng_act = jax.random.split(rng_act, env.num_agents)
actions = {agent: env.action_space(agent).sample(rng_act[i]) for i, agent in enumerate(env.agents)}

# Step environment
obs, states, rewards, dones, infos = env.step(rng_step, states, actions)
```

## Setup
To get started with using the environment please install all needed dependencies using:
```sh
pip install -r requirements.txt
```

## Training Baselines
The `baselines` directory provides training scripts needed to evaluate IPPO, MAPPO and PQN against the Craftax-MA and Craftax-Coop environments.

To use, following the steps below:
- Setup your environment according to the steps in the `SETUP` section
- Create a config yaml file and place in the `baselines/config` directory 
  - Default configurations for experiments are already provided
  - Make sure to modify the WandB information for appropriate logging
- Run one of the provided training scripts using the command
```sh
python baselines/<training-script> --config_file=<config-file-name>
```

## License
Code is licenced under the MIT license provided in the `LICENSE` document.
