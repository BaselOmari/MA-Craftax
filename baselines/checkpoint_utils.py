"""Orbax-based checkpoint and resume helpers for the training loop."""
import hashlib
import json
import os
import signal
import warnings

import jax
import orbax.checkpoint as ocp


_STOP = {"requested": False}


def sanitize_path_component(value, fallback="run"):
    """Return a filesystem-safe version of value, or fallback if empty."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(value).strip()
    ).strip("._-")
    return cleaned or fallback


def default_checkpoint_dir(config):
    """Return the stable absolute checkpoint directory derived from config."""
    configured = config.get("CHECKPOINT_DIR")
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    root = os.path.expanduser(config.get("OUTPUT_DIR", "") or ".")
    run = sanitize_path_component(config.get("RUN_NAME", "run"))
    return os.path.abspath(os.path.join(root, run, "checkpoints"))


def normalize_resume_mode(value):
    """Normalize a RESUME config value to one of 'auto', 'true', or 'false'."""
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if text not in {"auto", "true", "false"}:
        raise ValueError(f"RESUME must be 'auto', true, or false, got {value!r}.")
    return text


def sidecar_path(ckpt_dir):
    """Return the path of the run metadata file placed beside the checkpoint directory."""
    return os.path.join(os.path.dirname(os.path.abspath(ckpt_dir)), "run_meta.json")


def read_sidecar(ckpt_dir):
    """Read run metadata written beside the checkpoint directory, or None if absent."""
    path = sidecar_path(ckpt_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_sidecar(ckpt_dir, data):
    """Atomically write run metadata beside the checkpoint directory."""
    path = sidecar_path(ckpt_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def make_manager(ckpt_dir, config):
    """Create an orbax CheckpointManager for the given directory."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=int(config.get("CHECKPOINT_MAX_TO_KEEP", 3)),
        create=True,
        enable_async_checkpointing=True,
    )
    return ocp.CheckpointManager(ckpt_dir, options=options)


def structural_fingerprint(pytree):
    """Return a process-stable sha256 of the pytree leaf paths, shapes, and dtypes."""
    leaves_with_paths = jax.tree_util.tree_flatten_with_path(pytree)[0]
    items = []
    for path, leaf in leaves_with_paths:
        shape = tuple(getattr(leaf, "shape", ()))
        dtype = str(getattr(leaf, "dtype", type(leaf).__name__))
        items.append(f"{jax.tree_util.keystr(path)}|{shape}|{dtype}")
    blob = ";".join(items)
    return hashlib.sha256(blob.encode()).hexdigest()


def check_compatibility(meta, fingerprint, config):
    """Raise on shape or block-mapping mismatch; warn on learning-rate schedule change."""
    saved_fp = meta.get("fingerprint")
    if saved_fp != fingerprint:
        raise ValueError(
            "Cannot resume: checkpoint pytree structure does not match the current "
            f"config. Saved fingerprint {saved_fp}, current {fingerprint}. This means "
            "network size, agent count, or env dimensions changed. Start a fresh run "
            "or restore the original config."
        )
    saved_lui = meta.get("logging_updates_interval")
    if saved_lui is not None and saved_lui != config["LOGGING_UPDATES_INTERVAL"]:
        raise ValueError(
            "Cannot resume: LOGGING_UPDATES_INTERVAL changed from "
            f"{saved_lui} to {config['LOGGING_UPDATES_INTERVAL']}, which breaks the "
            "block-to-update-step mapping used to resume."
        )
    saved_nu = meta.get("num_updates")
    if saved_nu is not None and saved_nu != config["NUM_UPDATES"]:
        warnings.warn(
            "Resuming with a different NUM_UPDATES "
            f"({saved_nu} -> {config['NUM_UPDATES']}); the annealed learning-rate "
            "trajectory will differ from the original run.",
            stacklevel=2,
        )


def _nonzero_leaves(tree):
    """Return the pytree leaves that hold at least one element."""
    return [leaf for leaf in jax.tree_util.tree_leaves(tree) if getattr(leaf, "size", 1) != 0]


def save_checkpoint(mngr, carry, blocks_done, ckpt_dir, wandb_run_id, fingerprint, config, final=False):
    """Save the carry at its cumulative update step and update the sidecar, idempotently."""
    step = int(carry[1])
    if step in mngr.all_steps():
        return
    mngr.save(step, args=ocp.args.StandardSave({"leaves": _nonzero_leaves(carry)}))
    write_sidecar(
        ckpt_dir,
        {
            "wandb_run_id": wandb_run_id,
            "fingerprint": fingerprint,
            "blocks_done": int(blocks_done),
            "update_steps": step,
            "logging_updates_interval": config["LOGGING_UPDATES_INTERVAL"],
            "num_updates": config["NUM_UPDATES"],
            "final": bool(final),
        },
    )


def restore_carry(mngr, template_carry):
    """Restore the latest checkpoint into the structure of template_carry.

    Zero-size leaves are not stored (they carry no data) and are taken from the
    template, which has identical shapes.
    """
    latest = mngr.latest_step()
    leaves, treedef = jax.tree_util.tree_flatten(template_carry)
    template_nonzero = [leaf for leaf in leaves if getattr(leaf, "size", 1) != 0]
    restored = mngr.restore(latest, args=ocp.args.StandardRestore({"leaves": template_nonzero}))
    restored_iter = iter(restored["leaves"])
    merged = [leaf if getattr(leaf, "size", 1) == 0 else next(restored_iter) for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, merged)


def install_signal_handler():
    """Install SIGTERM and SIGUSR1 handlers that request a graceful stop."""
    def handler(signum, frame):
        _STOP["requested"] = True

    for sig in (signal.SIGTERM, signal.SIGUSR1):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def stop_requested():
    """Return True if a graceful stop has been requested via a signal."""
    return _STOP["requested"]
