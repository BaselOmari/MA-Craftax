import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import pytest
import yaml

import checkpoint_utils as ckpt

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1", reason="set RUN_E2E=1 to run the subprocess end-to-end test"
)

BASELINES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASELINES_DIR, "config", "test_checkpointing.yaml")


def _base_cfg():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["WANDB_MODE"] = "disabled"
    cfg["SAVE_VIDEO"] = False
    return cfg


def _write_cfg(path, cfg):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)


def _run(cfg_path):
    runner = textwrap.dedent(
        f"""
        import sys, yaml
        sys.path.insert(0, {BASELINES_DIR!r})
        import seperate_ippo_rnn as m
        with open({cfg_path!r}) as f:
            cfg = yaml.safe_load(f)
        m.single_run(cfg)
        """
    )
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    result = subprocess.run(
        [sys.executable, "-c", runner],
        env=env,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    if result.returncode != 0:
        raise AssertionError(f"run failed:\nSTDOUT\n{result.stdout}\nSTDERR\n{result.stderr[-4000:]}")
    return result


def _final_carry(cfg, ckpt_dir):
    import seperate_ippo_rnn as m

    env = m.build_env(cfg)
    init_carry, _, _ = m.make_train(cfg, env)(jax.random.PRNGKey(cfg["SEED"]))
    mngr = ckpt.make_manager(ckpt_dir, cfg)
    carry = ckpt.restore_carry(mngr, init_carry)
    mngr.close()
    return carry


def test_cli_resume_matches_uninterrupted(tmp_path):
    dir_a = str(tmp_path / "a_ckpt")
    dir_b = str(tmp_path / "b_ckpt")

    cfg_a = _base_cfg()
    cfg_a["CHECKPOINT_DIR"] = dir_a
    cfg_a["OUTPUT_DIR"] = str(tmp_path / "a_out")
    cfg_a["RESUME"] = False
    path_a = str(tmp_path / "cfg_a.yaml")
    _write_cfg(path_a, cfg_a)
    _run(path_a)

    k = max(1, cfg_a["NUM_LOGGING_ITERS"] // 2)
    cfg_b1 = _base_cfg()
    cfg_b1["CHECKPOINT_DIR"] = dir_b
    cfg_b1["OUTPUT_DIR"] = str(tmp_path / "b_out")
    cfg_b1["RESUME"] = "auto"
    cfg_b1["MAX_BLOCKS_THIS_RUN"] = k
    path_b1 = str(tmp_path / "cfg_b1.yaml")
    _write_cfg(path_b1, cfg_b1)
    _run(path_b1)

    mngr_b = ckpt.make_manager(dir_b, cfg_b1)
    assert mngr_b.latest_step() == k * cfg_b1["LOGGING_UPDATES_INTERVAL"]
    mngr_b.close()

    cfg_b2 = dict(cfg_b1)
    cfg_b2["MAX_BLOCKS_THIS_RUN"] = 0
    path_b2 = str(tmp_path / "cfg_b2.yaml")
    _write_cfg(path_b2, cfg_b2)
    _run(path_b2)

    carry_a = _final_carry(cfg_a, dir_a)
    carry_b = _final_carry(cfg_b2, dir_b)

    assert int(carry_a[1]) == cfg_a["NUM_UPDATES"]
    assert int(carry_b[1]) == cfg_b2["NUM_UPDATES"]

    leaves_a = jax.tree_util.tree_leaves(carry_a)
    leaves_b = jax.tree_util.tree_leaves(carry_b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b):
        assert jnp.allclose(x, y, atol=1e-5, rtol=1e-5), "resumed run diverged from uninterrupted run"
