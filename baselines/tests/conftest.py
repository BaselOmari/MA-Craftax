import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("WANDB_MODE", "disabled")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
