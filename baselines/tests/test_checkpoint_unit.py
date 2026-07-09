import os
import warnings

import jax
import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState

import orbax.checkpoint as ocp

import checkpoint_utils as ckpt


NUM_AGENTS = 3
HID = 4


def _linear_schedule(count):
    return 0.001 * (1.0 - count / 100.0)


def _make_tx():
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=_linear_schedule, eps=1e-5),
    )


def _make_train_state(seed):
    def init_one(r):
        return {"kernel": jax.random.normal(r, (HID,))}

    params = jax.vmap(init_one)(jax.random.split(jax.random.PRNGKey(seed), NUM_AGENTS))

    def apply_fn(variables, x):
        return x

    return TrainState.create(apply_fn=apply_fn, params=params, tx=_make_tx())


def _tree_equal(a, b):
    return jax.tree_util.tree_all(
        jax.tree.map(lambda x, y: bool(jnp.array_equal(x, y)), a, b)
    )


def _roundtrip(pytree, template, tmp_path, step=0):
    mngr = ocp.CheckpointManager(
        str(tmp_path / "ck"),
        options=ocp.CheckpointManagerOptions(create=True),
    )
    mngr.save(step, args=ocp.args.StandardSave(pytree))
    mngr.wait_until_finished()
    out = mngr.restore(step, args=ocp.args.StandardRestore(template))
    mngr.close()
    return out


def test_orbax_roundtrip_bit_exact(tmp_path):
    ts = _make_train_state(0)
    grads = jax.tree.map(lambda p: jnp.ones_like(p), ts.params)
    ts = ts.apply_gradients(grads=grads).apply_gradients(grads=grads)

    payload = {
        "carry": (
            (ts, {"a": jnp.arange(4), "b": jnp.ones((2, 2))}, jax.random.PRNGKey(7)),
            jnp.asarray(1234, dtype=jnp.int32),
        )
    }
    template = {
        "carry": (
            (_make_train_state(999), {"a": jnp.zeros(4, jnp.int32), "b": jnp.zeros((2, 2))}, jax.random.PRNGKey(0)),
            jnp.asarray(0, dtype=jnp.int32),
        )
    }
    restored = _roundtrip(payload, template, tmp_path)

    r_ts = restored["carry"][0][0]
    assert _tree_equal(r_ts.params, ts.params)
    assert _tree_equal(r_ts.opt_state, ts.opt_state)
    assert int(r_ts.step) == int(ts.step)
    assert int(restored["carry"][1]) == 1234
    assert restored["carry"][1].dtype == jnp.int32
    assert bool(jnp.array_equal(restored["carry"][0][2], payload["carry"][0][2]))
    assert int(r_ts.apply_gradients(grads=grads).step) == int(ts.step) + 1


def test_lr_schedule_resumes(tmp_path):
    tx = _make_tx()
    params = {"w": jnp.ones((5,))}
    grad = {"w": jnp.full((5,), 0.3)}
    state = tx.init(params)
    for _ in range(7):
        _, state = tx.update(grad, state, params)

    restored_state = _roundtrip(state, tx.init(params), tmp_path)

    u_saved, _ = tx.update(grad, state, params)
    u_restored, _ = tx.update(grad, restored_state, params)
    u_fresh, _ = tx.update(grad, tx.init(params), params)

    assert _tree_equal(u_saved, u_restored)
    assert not _tree_equal(u_restored, u_fresh)


def test_fingerprint_stable_and_sensitive():
    tree_a = {"x": jnp.zeros((2, 3)), "y": jnp.zeros((4,), jnp.int32)}
    tree_a2 = {"x": jnp.ones((2, 3)), "y": jnp.ones((4,), jnp.int32)}
    tree_b = {"x": jnp.zeros((2, 4)), "y": jnp.zeros((4,), jnp.int32)}

    fp_a = ckpt.structural_fingerprint(tree_a)
    assert fp_a == ckpt.structural_fingerprint(tree_a2)
    assert fp_a != ckpt.structural_fingerprint(tree_b)


def test_fingerprint_ignores_static_train_state_fields():
    fp1 = ckpt.structural_fingerprint(_make_train_state(0))
    fp2 = ckpt.structural_fingerprint(_make_train_state(1))
    assert fp1 == fp2


def test_sidecar_atomic_roundtrip(tmp_path):
    ckpt_dir = str(tmp_path / "run" / "checkpoints")
    data = {"wandb_run_id": "abc", "fingerprint": "deadbeef", "blocks_done": 3, "update_steps": 6}
    ckpt.write_sidecar(ckpt_dir, data)

    path = ckpt.sidecar_path(ckpt_dir)
    assert path == str(tmp_path / "run" / "run_meta.json")
    assert ckpt.read_sidecar(ckpt_dir) == data
    assert not any(p.name.endswith(".tmp") for p in (tmp_path / "run").iterdir())


def test_read_sidecar_missing(tmp_path):
    assert ckpt.read_sidecar(str(tmp_path / "nope" / "checkpoints")) is None


def test_default_checkpoint_dir(tmp_path):
    cfg = {"OUTPUT_DIR": str(tmp_path), "RUN_NAME": "My Run/01"}
    d = ckpt.default_checkpoint_dir(cfg)
    assert os.path.isabs(d)
    assert d == str(tmp_path / "My_Run_01" / "checkpoints")

    cfg2 = {"CHECKPOINT_DIR": str(tmp_path / "explicit")}
    assert ckpt.default_checkpoint_dir(cfg2) == str(tmp_path / "explicit")


def test_check_compatibility():
    base = {
        "fingerprint": "fp",
        "logging_updates_interval": 2,
        "num_updates": 11,
    }
    cfg = {"LOGGING_UPDATES_INTERVAL": 2, "NUM_UPDATES": 11}

    ckpt.check_compatibility(base, "fp", cfg)

    with pytest.raises(ValueError):
        ckpt.check_compatibility(base, "different_fp", cfg)

    with pytest.raises(ValueError):
        ckpt.check_compatibility(base, "fp", {"LOGGING_UPDATES_INTERVAL": 4, "NUM_UPDATES": 11})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ckpt.check_compatibility(base, "fp", {"LOGGING_UPDATES_INTERVAL": 2, "NUM_UPDATES": 22})
        assert any("learning-rate" in str(w.message) for w in caught)


def test_keep_last_n_and_idempotent_save(tmp_path):
    cfg = {
        "CHECKPOINT_MAX_TO_KEEP": 2,
        "LOGGING_UPDATES_INTERVAL": 2,
        "NUM_UPDATES": 11,
    }
    ckpt_dir = str(tmp_path / "run" / "checkpoints")
    mngr = ckpt.make_manager(ckpt_dir, cfg)

    def carry_at(step):
        return (({"p": jnp.ones((2,)) * step},), jnp.asarray(step, dtype=jnp.int32))

    for step in (2, 4, 6):
        ckpt.save_checkpoint(mngr, carry_at(step), step // 2, ckpt_dir, "wid", "fp", cfg)
    mngr.wait_until_finished()
    assert sorted(mngr.all_steps()) == [4, 6]
    assert mngr.latest_step() == 6

    steps_before = sorted(mngr.all_steps())
    ckpt.save_checkpoint(mngr, carry_at(6), 3, ckpt_dir, "wid", "fp", cfg)
    mngr.wait_until_finished()
    assert sorted(mngr.all_steps()) == steps_before
    mngr.close()


def test_save_restore_handles_zero_size_arrays(tmp_path):
    cfg = {"CHECKPOINT_MAX_TO_KEEP": 2, "LOGGING_UPDATES_INTERVAL": 2, "NUM_UPDATES": 11}
    ckpt_dir = str(tmp_path / "run" / "checkpoints")
    mngr = ckpt.make_manager(ckpt_dir, cfg)

    carry = (
        (jnp.ones((2, 3)), jnp.zeros((0, 4)), {"m": jnp.zeros((0,)), "n": jnp.arange(5)}),
        jnp.asarray(6, dtype=jnp.int32),
    )
    ckpt.save_checkpoint(mngr, carry, 3, ckpt_dir, "wid", "fp", cfg)
    mngr.wait_until_finished()

    template = (
        (jnp.zeros((2, 3)), jnp.zeros((0, 4)), {"m": jnp.zeros((0,)), "n": jnp.zeros((5,), jnp.int32)}),
        jnp.asarray(0, dtype=jnp.int32),
    )
    restored = ckpt.restore_carry(mngr, template)
    mngr.close()

    assert _tree_equal(restored, carry)
    assert restored[0][1].shape == (0, 4)
    assert int(restored[1]) == 6


def test_stop_requested_toggle():
    ckpt._STOP["requested"] = False
    assert ckpt.stop_requested() is False
    ckpt._STOP["requested"] = True
    assert ckpt.stop_requested() is True
    ckpt._STOP["requested"] = False
