import os

import jax
import jax.numpy as jnp
import pytest
import yaml

import checkpoint_utils as ckpt


def _load_cfg():
    path = os.path.join(os.path.dirname(__file__), "..", "config", "test_checkpointing.yaml")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["WANDB_MODE"] = "disabled"
    cfg["SAVE_VIDEO"] = False
    cfg["EARLY_EPISODE_CAP"] = 50
    cfg["EARLY_EPISODE_CAP_UNTIL"] = 6
    cfg["GENERAL_EPISODE_CAP"] = 0
    return cfg


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    import wandb

    import seperate_ippo_rnn as m

    cfg = _load_cfg()
    cfg["OUTPUT_DIR"] = str(tmp_path_factory.mktemp("out"))
    wandb.init(mode="disabled", config=cfg)
    env = m.build_env(cfg)
    train_fn = m.make_train(cfg, env)
    init_carry, update_plot_fn, update_step_fn = train_fn(jax.random.PRNGKey(cfg["SEED"]))
    jit_plot = jax.jit(update_plot_fn)
    yield {
        "cfg": cfg,
        "init_carry": init_carry,
        "jit_plot": jit_plot,
        "update_step_fn": update_step_fn,
        "fingerprint": ckpt.structural_fingerprint(init_carry),
    }
    wandb.finish()


def _run_blocks(jit_plot, carry, start, end):
    for _ in range(start, end):
        carry, _ = jit_plot(carry, None)
    return jax.block_until_ready(carry)


def _run_tail(update_step_fn, cfg, carry):
    if cfg["REMAINING_UPDATES"] > 0 and int(carry[1]) < cfg["NUM_UPDATES"]:
        scan_tail = jax.jit(
            lambda c: jax.lax.scan(update_step_fn, c, None, cfg["REMAINING_UPDATES"])
        )
        carry, _ = scan_tail(carry)
    return jax.block_until_ready(carry)


def _bit_exact(a, b):
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    return all(bool(jnp.array_equal(x, y)) for x, y in zip(leaves_a, leaves_b))


def test_config_has_tail_and_curriculum_flip(runtime):
    cfg = runtime["cfg"]
    assert cfg["NUM_LOGGING_ITERS"] >= 4
    assert cfg["REMAINING_UPDATES"] >= 1
    assert cfg["EARLY_EPISODE_CAP_UNTIL"] < cfg["NUM_UPDATES"]


def test_resume_equals_uninterrupted(runtime, tmp_path):
    cfg = runtime["cfg"]
    jit_plot = runtime["jit_plot"]
    update_step_fn = runtime["update_step_fn"]
    init_carry = runtime["init_carry"]
    fingerprint = runtime["fingerprint"]
    n = cfg["NUM_LOGGING_ITERS"]
    k = n // 2

    carry = _run_blocks(jit_plot, init_carry, 0, n)
    full = _run_tail(update_step_fn, cfg, carry)

    ckpt_dir = str(tmp_path / "run" / "checkpoints")
    mngr = ckpt.make_manager(ckpt_dir, cfg)
    carry_k = _run_blocks(jit_plot, init_carry, 0, k)
    ckpt.save_checkpoint(mngr, carry_k, k, ckpt_dir, "wid", fingerprint, cfg)
    mngr.wait_until_finished()

    restored = ckpt.restore_carry(mngr, init_carry)
    mngr.close()
    assert int(restored[1]) == k * cfg["LOGGING_UPDATES_INTERVAL"]
    assert _bit_exact(restored, carry_k)

    carry_r = _run_blocks(jit_plot, restored, k, n)
    resumed = _run_tail(update_step_fn, cfg, carry_r)

    assert int(full[1]) == cfg["NUM_UPDATES"]
    assert int(resumed[1]) == cfg["NUM_UPDATES"]
    assert _bit_exact(full, resumed)


def test_resume_from_every_block_boundary(runtime, tmp_path):
    cfg = runtime["cfg"]
    jit_plot = runtime["jit_plot"]
    init_carry = runtime["init_carry"]
    fingerprint = runtime["fingerprint"]
    n = cfg["NUM_LOGGING_ITERS"]

    reference = _run_blocks(jit_plot, init_carry, 0, n)

    for k in range(1, n):
        ckpt_dir = str(tmp_path / f"k{k}" / "checkpoints")
        mngr = ckpt.make_manager(ckpt_dir, cfg)
        carry_k = _run_blocks(jit_plot, init_carry, 0, k)
        ckpt.save_checkpoint(mngr, carry_k, k, ckpt_dir, "wid", fingerprint, cfg)
        mngr.wait_until_finished()
        restored = ckpt.restore_carry(mngr, init_carry)
        mngr.close()
        assert int(restored[1]) // cfg["LOGGING_UPDATES_INTERVAL"] == k
        resumed = _run_blocks(jit_plot, restored, k, n)
        assert _bit_exact(reference, resumed)


def test_resume_when_finished_is_noop(runtime):
    cfg = runtime["cfg"]
    jit_plot = runtime["jit_plot"]
    update_step_fn = runtime["update_step_fn"]
    init_carry = runtime["init_carry"]
    n = cfg["NUM_LOGGING_ITERS"]

    carry = _run_blocks(jit_plot, init_carry, 0, n)
    carry = _run_tail(update_step_fn, cfg, carry)

    start_block = int(carry[1]) // cfg["LOGGING_UPDATES_INTERVAL"]
    assert start_block >= n
    tail_needed = cfg["REMAINING_UPDATES"] > 0 and int(carry[1]) < cfg["NUM_UPDATES"]
    assert not tail_needed


def test_restore_shape_mismatch_raises(runtime, tmp_path):
    cfg = runtime["cfg"]
    jit_plot = runtime["jit_plot"]
    init_carry = runtime["init_carry"]
    fingerprint = runtime["fingerprint"]

    ckpt_dir = str(tmp_path / "run" / "checkpoints")
    mngr = ckpt.make_manager(ckpt_dir, cfg)
    ckpt.save_checkpoint(mngr, init_carry, 0, ckpt_dir, "wid", fingerprint, cfg)
    mngr.wait_until_finished()

    runner_state, update_steps = init_carry
    bad_hstate = jnp.zeros(runner_state[4].shape[:-1] + (runner_state[4].shape[-1] + 1,))
    bad_runner = (runner_state[0], runner_state[1], runner_state[2], runner_state[3], bad_hstate, runner_state[5])
    bad_template = (bad_runner, update_steps)

    with pytest.raises(Exception):
        ckpt.restore_carry(mngr, bad_template)
    mngr.close()
