from typing import Any, Dict, Set, Tuple
from mlff.cAPI.process_argparse import StoreDictKeyPair
import numpy as np
import jax
import jax.numpy as jnp
import os
import wandb
import portpicker
import ase.units as si

from mlff.io.io import create_directory, bundle_dicts, save_dict
from mlff.training import Coach, Optimizer, get_loss_fn, create_train_state
from mlff.data import DataTuple, DataSet

from mlff.nn.stacknet import (
    get_obs_and_force_fn,
    get_observable_fn,
    get_energy_force_stress_fn,
)
from mlff.nn import So3krates
from mlff.properties import (
    shnitsel_property_keys_dynamic as prop_keys_shnitsel_dynamic,
    shnitsel_property_keys_static as prop_keys_shnitsel_static,
)
from mlff.properties import property_names

import mlff.properties.property_names as pn
from netCDF4 import Dataset
import sys

import shnitsel as sh
import xarray as xr
from shnitsel.core.parse.common import transform_atom_name_to_number
import pathlib
import shnitsel.xarray
import logging

import argparse

from rdkit import Chem
import glob

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def import_shnitsel_dynamic(
    data_path: str,
    prop_keys,
) -> None | Tuple[int, Dict, Dict, xr.Dataset]:
    pass


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Analyze the inference results and compare with targets."
    )

    # Add the arguments
    # parser.add_argument(
    #    "--ckpt_dir",
    #    type=str,
    #    required=True,
    #    default=os.getcwd(),
    #    help="Path to the checkpoint directory.",
    # )

    # Add the arguments
    parser.add_argument(
        "--inference_path",
        type=str,
        required=True,
        default=os.getcwd(),
        help="Path to the npz archive saved by inference.",
    )

    parser.add_argument(
        "--units",
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1,KEY2=VAL2...",
        default=None,
        help="Units in the data set for the quantities. Needs only to be specified"
        "if the model has been trained on units different from the ones present in the data set.",
    )

    parser.add_argument("--targets", nargs="+", required=False, default=None)

    args = parser.parse_args()

    # Read arguments
    # ckpt_dir = args.ckpt_dir
    inference_path = args.inference_path
    units = args.units

    inference_results = np.load(inference_path, allow_pickle=True)

    inputs = inference_results["inputs"].item()
    predictions = inference_results["predictions"].item()
    targets = inference_results["targets"].item()
    cm_inferno = cm.get_cmap("inferno")

    atomic_states_atom = inputs["atomic_state"].reshape(-1)
    atomic_states_mol = np.max(inputs["atomic_state"], axis=1)
    color_atom = cm_inferno(atomic_states_atom / 4)
    color_mol = cm_inferno(atomic_states_mol / 4)

    # print(targets)
    # print(type(targets))
    # print(predictions)
    # print(type(predictions))
    # import json

    # with open(ckpt_dir + "/splits.json", "r") as splitin:
    # splits = json.load(splitin)["split"]
    # print(list(splits.keys()))
    # print(len(splits["data_idx_train"]))
    # print(len(splits["data_idx_valid"]))
    # print(len(splits["data_idx_test"]))

    # test_ids = splits["data_idx_test"]
    plt.clf()

    fig, (ax_energy, ax_force) = plt.subplots(1, 2)

    ax_energy.scatter(
        targets["energy"],
        predictions["energy"],
        rasterized=True,
        color=color_mol,
    )

    max_e = max(np.max(
        targets["energy"]),
        np.max(predictions["energy"]))

    min_e = min(np.min(
        targets["energy"]),
        np.min(predictions["energy"]))

    ax_energy.plot([min_e, max_e], [min_e, max_e], color="grey", ls="dashed")

    ax_energy.set_xlabel("$E_{target}$ [eV]")
    ax_energy.set_ylabel("$E_{predict}$ [eV]")

    target_force = targets["force"]
    predictions_force = predictions["force"]

    target_force_norm = np.linalg.norm(target_force, axis=2, ord=2).reshape(-1)
    predictions_force_norm = np.linalg.norm(predictions_force, axis=2, ord=2).reshape(
        -1
    )

    ax_force.scatter(
        target_force_norm,
        predictions_force_norm,
        rasterized=True,
        color=color_atom,
    )

    max_f = max(np.max(
        target_force_norm),
        np.max(predictions_force_norm))

    min_f = min(np.min(
        target_force_norm),
        np.min(predictions_force_norm))

    ax_force.plot([min_f, max_f], [min_f, max_f], color="grey", ls="dashed")

    ax_force.set_xlabel("$|F_{target}|$ [?]")
    ax_force.set_ylabel("$|F_{predict}|$ [?]")

    fig.tight_layout()
    fig.savefig(inference_path + "_target_relation.pdf")
