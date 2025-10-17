from typing import Dict, Any
from pathlib import Path
import os

from orbax.checkpoint import PyTreeCheckpointer, Checkpointer, PyTreeCheckpointHandler
import orbax.checkpoint as ocp
import pathlib

__STEP_PREFIX__: str = "ckpt"


def load_params_from_ckpt_dir(ckpt_dir):
    try:
        return load_state_from_ckpt_dir(ckpt_dir)["valid_params"]
    except ValueError:
        try:
            registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
            registry.add(
                "state", ocp.args.StandardRestore, ocp.StandardCheckpointHandler
            )
            options = ocp.CheckpointManagerOptions(step_prefix=__STEP_PREFIX__)

            with ocp.CheckpointManager(
                pathlib.Path(ckpt_dir).resolve(),
                handler_registry=registry,
                options=options,
            ) as mngr:
                mngr_state = mngr.restore(mngr.latest_step())

                state = mngr_state.get("state")

                return state["valid_params"]
        except ValueError:
            raise RuntimeError(
                f"Loading model parameters from checkpoint saved at {ckpt_dir} failed. "
                "This error typically occurs if within the ckpt_XXX directory there is another folder. "
                "Consider moving the folder somewhere else."
            )


def load_state_from_ckpt_dir(ckpt_dir: str):
    # mngr = CheckpointManager(ckpt_dir, __CHECKPOINTERS__, options=CheckpointManagerOptions(step_prefix=__STEP_PREFIX__))
    # return mngr.restore(n)['state']

    ns = []
    abs_ckpt_dir = Path(ckpt_dir).resolve().absolute()
    for u in os.scandir(abs_ckpt_dir):
        if u.is_dir():
            dir_name = Path(u).stem
            prefix_n = dir_name.split("_")
            if len(prefix_n) == 2:
                prefix, n = prefix_n
                if prefix == __STEP_PREFIX__:
                    ns += [int(n)]
    max_step = max(ns)

    registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
    registry.add("state", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
    options = ocp.CheckpointManagerOptions(step_prefix=__STEP_PREFIX__)

    with ocp.CheckpointManager(
        pathlib.Path(ckpt_dir).resolve(),
        handler_registry=registry,
        options=options,
    ) as mngr:
        # mngr.restore(
        #    max_step, args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract_pytree))
        # )
        cpt = mngr.restore(max_step)

        return cpt["state"]
    return None
    # ckptr = Checkpointer(PyTreeCheckpointHandler())
    # return ckptr.restore(
    #    abs_ckpt_dir / f"{__STEP_PREFIX__}_{max_step}/state", item=None
    # )


def _load_params_from_ckpt_dir(ckpt_dir: str):
    return load_state_from_ckpt_dir(ckpt_dir)["valid_params"]
