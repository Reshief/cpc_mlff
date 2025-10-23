from typing import Any, Dict, Set, Tuple
from mlff.cAPI.process_argparse import StoreDictKeyPair
import numpy as np
import jax
import jax.numpy as jnp
import os
import wandb
import portpicker
import ase.units as si

from mlff.io.checkpoint import load_params_from_ckpt_dir
from mlff.io.io import create_directory, bundle_dicts, read_json, save_dict
from mlff.nn.embed.embed import AtomTypeEmbed, MolecularStateEmbed
from mlff.nn.stacknet.stacknet import init_stack_net
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


def get_full_system_list(data_dir: pathlib.Path):
    systems = dict()
    root_dir = data_dir.resolve().as_posix()

    for entry in glob.glob(
        "./**/*.nc",
        root_dir=root_dir,
        recursive=True,
    ):
        entry = data_dir / entry
        if entry.is_file():
            name_parts = entry.stem.split("_")
            system_id = name_parts[0]
            is_static = name_parts[-1] == "static"
            if system_id not in systems.keys():
                systems[system_id] = []

            systems[system_id].append(
                {
                    "system_id": system_id,
                    "static": is_static,
                    "path": entry.resolve().as_posix(),
                }
            )
    return systems


def load_system_data(
    systems_list: Dict[str, Dict], use_dynamic_traj=True, use_static_traj=False
):

    system_ids = sorted(systems_list.keys())

    total_loaded_data = []

    n_data_total = 0

    for id in system_ids:
        system_files = systems_list[id]
        system_files = sorted(system_files, key=lambda x: x["path"])

        # print(system_files)

        for entry in system_files:
            if entry["static"] and use_static_traj:
                res = import_shnitsel_static(
                    entry["path"],
                    prop_keys_shnitsel_static,
                )
                if res is None:
                    continue
                n_data, prop_keys, data_arrays, dataset = res
            elif not entry["static"] and use_dynamic_traj:
                res = import_shnitsel_dynamic(
                    entry["path"], prop_keys_shnitsel_dynamic)
                if res is None:
                    continue
                n_data, prop_keys, data_arrays, dataset = res
            else:
                continue

            n_data_total += n_data
            total_loaded_data.append(
                {
                    "system_id": entry["system_id"],
                    "static": entry["static"],
                    "path": entry["path"],
                    "n_data": n_data,
                    "prop_keys": prop_keys,
                    "data_as_array": data_arrays,
                    "dataset": dataset,
                }
            )

    return n_data_total, total_loaded_data


def merge_array_across_systems(
    key: str, loaded_datasets, fillValue: Any = 0, repeat_scalars: bool = True
):
    logging.info(f"Merging key: <{key}>")
    max_length = 0
    for entry in loaded_datasets:
        sys_prop_keys = entry["prop_keys"]
        data_array = entry["data_as_array"][sys_prop_keys[key]]

        max_length = max(
            max_length,
            (1 if len(data_array.shape) < 2 else data_array.shape[1]),
        )

    logging.info(f"Max length of  <{key}> is {max_length}")

    merged_array = None
    is_scalar = False
    dtype = None
    for entry in loaded_datasets:
        sys_prop_keys = entry["prop_keys"]
        data_array = entry["data_as_array"][sys_prop_keys[key]]

        data_shape = np.array(data_array.shape)
        dtype = data_array.dtype

        if len(data_shape) < 2:

            if merged_array is None:
                merged_array = data_array
                is_scalar = True
            else:
                if not is_scalar:
                    raise RuntimeError(
                        f"Trying to merge scalar array with non-scalar array for different systems at key {key}"
                    )
                else:
                    merged_array = np.append(merged_array, data_array, axis=0)
            # raise NotImplementedError("Merging of scalar data not yet implemented")
        else:
            if is_scalar:
                raise RuntimeError(
                    f"Trying to merge scalar array with non-scalar array for different systems at key {key}"
                )
            extension_needed = max_length - data_shape[1]

            if extension_needed > 0:
                target_shape = data_shape
                target_shape[1] = extension_needed
                data_array = np.append(
                    data_array, np.full(target_shape, fillValue, dtype=dtype), axis=1
                )

            if merged_array is None:
                merged_array = data_array
            else:
                merged_array = np.append(merged_array, data_array, axis=0)

    return key, merged_array


def merge_system_data_set(loaded_datasets) -> Tuple[int, Dict, Dict]:

    initial_keyset_intersection: None | Set = None

    initial_key_dtype = dict()

    for entry in loaded_datasets:
        sys_prop_keys = entry["prop_keys"]
        data_array = entry["data_as_array"]

        available_set = set()

        for key, val in sys_prop_keys.items():
            if key in data_array:
                available_set.add(key)
                initial_key_dtype[key] = data_array[key].dtype.type

        if initial_keyset_intersection is None:
            initial_keyset_intersection = available_set
        else:
            initial_keyset_intersection = initial_keyset_intersection.intersection(
                available_set
            )

    logging.info(
        f"Keyset intersection across all data sets: {initial_keyset_intersection}"
    )

    if initial_keyset_intersection is None:
        raise RuntimeError("No intersection in system keys")

    final_keyset = dict()
    final_data_arrays = dict()
    final_n_data = 0

    for shared_key in initial_keyset_intersection:
        key, merged_array = merge_array_across_systems(
            shared_key,
            loaded_datasets,
            fillValue=get_fill_value_for(initial_key_dtype[shared_key]),
        )
        print(
            "Mapped type:",
            shared_key,
            "(shape:",
            merged_array.shape,
            ")",
            initial_key_dtype[shared_key],
            get_fill_value_for(initial_key_dtype[shared_key]),
        )
        final_keyset[key] = key
        final_data_arrays[key] = merged_array
        final_n_data = len(merged_array)

    return final_n_data, final_keyset, final_data_arrays


def get_fill_value_for(datatype: type):
    if issubclass(datatype, bool) or issubclass(datatype, np.bool):
        return False
    elif issubclass(datatype, int) or issubclass(datatype, np.integer):
        return int(0)
    elif issubclass(datatype, str) or issubclass(datatype, np.character):
        return ""
    elif issubclass(datatype, float) or issubclass(datatype, np.floating):
        return 0.0
    else:
        raise ValueError(
            f"No default filling value found for type: {datatype}")


def adjacency_from_mol(rdkit_mol) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    # print(repr(rdkit_mol))
    # print(rdkit_mol.GetNumBonds())
    am = Chem.GetAdjacencyMatrix(rdkit_mol)
    # print(am)
    # sys.exit(1)

    adj_i = []
    adj_j = []

    for i in range(len(am)):
        for j in range(len(am[i])):
            if am[i, j] != 0:
                adj_i.append(i)
                adj_j.append(j)

    num_pairs = len(adj_i)
    print(f"Found {num_pairs} edges")

    pair_mask = np.full((num_pairs,), True, dtype=np.bool)

    return (
        xr.DataArray(adj_i, dims=("pair_index",), name="idx_i"),
        xr.DataArray(adj_j, dims=("pair_index",), name="idx_j"),
        xr.DataArray(pair_mask, dims=("pair_index",), name="pair_mask"),
    )


def import_shnitsel_dynamic(
    data_path: str,
    prop_keys,
) -> None | Tuple[int, Dict, Dict, xr.Dataset]:
    logging.info(f"Importing trajectory from: {data_path}")
    atom_position_key = prop_keys[property_names.atomic_position]
    E_key = prop_keys[property_names.energy]
    F_key = prop_keys[property_names.force]
    atom_type_key = prop_keys[property_names.atomic_type]
    atomic_state_key = prop_keys[property_names.atomic_state]

    # Load dataset
    dataset: xr.Dataset = sh.open_frames(data_path)

    varkeys = list(dataset.variables.keys())
    if (
        atom_position_key not in varkeys
        or E_key not in varkeys
        or F_key not in varkeys
        or atom_type_key not in varkeys
        or atomic_state_key not in varkeys
    ):
        logging.warning(
            f"Trajectory {data_path} is missing one or more of the keys : {atom_position_key}, {atomic_state_key}, {atom_type_key}, {E_key}, {F_key}. It has keys <<{varkeys}>>. The trajectory will be skipped."
        )
        return None

    # Normalize units

    # data already in eV
    # dataset[E_key] = dataset[E_key]
    # convert data to eV from hartree/bohr used in Shnitsel
    print(dataset[F_key].units)
    print(dataset[E_key].units)
    # dataset[E_key].values *= si.Hartree
    # dataset[E_key].assign_attrs(units="eV")
    # dataset[atom_position_key].values *= si.Bohr
    # dataset[atom_position_key].assign_attrs(units="angstrom")
    dataset[F_key].values *= si.Hartree / si.Bohr
    dataset[F_key].assign_attrs(units="eV/angstrom")
    dataset[E_key].assign_attrs(units="eV")
    dataset[atom_position_key].assign_attrs(units="angstrom")

    # print(repr(dataset))
    # print(repr(dataset[E_key]))
    # print(repr(dataset[F_key]))
    print(
        "Energy (max/median/avg/min)",
        np.max(dataset[E_key].values),
        np.median(dataset[E_key].values),
        np.average(dataset[E_key].values),
        np.min(dataset[E_key].values),
    )

    forces_norm = np.linalg.norm(dataset[F_key].values, axis=2, ord=2)
    print(
        "Force (max/median/avg/min)",
        np.max(forces_norm),
        np.median(forces_norm),
        np.average(forces_norm),
        np.min(forces_norm),
    )

    final_prop_keys = {}
    final_prop_keys.update(prop_keys)

    varkeys = list(dataset.variables.keys())
    if E_key not in varkeys or F_key not in varkeys or atom_type_key not in varkeys:
        logging.warning(
            f"Trajectory {data_path} is missing one or more of the keys : {atom_type_key}, {E_key}, {F_key}. It has keys <<{varkeys}>>. The trajectory will be skipped."
        )
        return None

    # Restructure the dimensions into one continuous frame/state dimension

    lead_dimension = "frame"

    # print(repr(dataset))

    state_size = dataset.coords[atomic_state_key]
    # print("State coord:", repr(state_size.values.shape))
    num_states = state_size.values.shape[0]
    dataset = dataset.assign_coords({"state": np.arange(1, 1 + num_states)})

    dataset = dataset.transpose(lead_dimension, ...)
    if lead_dimension in dataset.indexes or lead_dimension in dataset.xindexes:
        dataset = dataset.reset_index(lead_dimension)
    dataset = dataset.stack(data=[lead_dimension, atomic_state_key])
    dataset = dataset.transpose("data", ...)
    lead_dimension = "data"

    # Translate atom types into number representations
    atom_type = dataset.variables[atom_type_key]
    atom_number = transform_atom_name_to_number(atom_type)

    rdkit_mol = dataset.isel(data=0).atXYZ.sh.to_mol()
    idx_i, idx_j, pair_mask = adjacency_from_mol(rdkit_mol)

    n_data = dataset.sizes[lead_dimension]
    n_atoms = dataset.sizes["atom"]

    # Make atom number array span the entirety of the dataset
    atom_number_array = xr.DataArray(
        atom_number).expand_dims({lead_dimension: n_data})
    atom_number_array = atom_number_array.transpose(lead_dimension, ...)

    atom_idx_i = idx_i.expand_dims({lead_dimension: n_data})
    atom_idx_i = atom_idx_i.transpose(lead_dimension, ...)
    atom_idx_j = idx_j.expand_dims({lead_dimension: n_data})
    atom_idx_j = atom_idx_j.transpose(lead_dimension, ...)
    atom_pair_mask = pair_mask.expand_dims({lead_dimension: n_data})
    atom_pair_mask = atom_pair_mask.transpose(lead_dimension, ...)

    # Create node masks
    node_mask = xr.DataArray(
        np.full((n_atoms,), True, dtype=bool), dims=("atom",), name="nodes_mask"
    ).expand_dims({lead_dimension: n_data})
    node_mask = node_mask.transpose(lead_dimension, ...)

    # Append new variables to dataset
    dataset = dataset.assign(atomic_type=atom_number_array)
    dataset = dataset.assign(idx_i=atom_idx_i)
    dataset = dataset.assign(idx_j=atom_idx_j)
    dataset = dataset.assign(pair_mask=atom_pair_mask)
    dataset = dataset.assign(node_mask=node_mask)
    final_prop_keys[property_names.atomic_type] = "atomic_type"

    # Make state into a per-atom variable
    molecule_state_array = dataset[atomic_state_key]
    per_atom_state_array = molecule_state_array.expand_dims({"atom": n_atoms})
    per_atom_state_array = per_atom_state_array.transpose(lead_dimension, ...)
    dataset = dataset.assign(atomic_state_tmp=per_atom_state_array)
    final_prop_keys[property_names.atomic_state] = "atomic_state_tmp"

    # print(repr(dataset))

    # print(repr(dataset["state"]))

    property_keys = dict()
    dataset_arrays = dict()

    rename_keys = dict()

    for key, value in final_prop_keys.items():
        if value in dataset.variables:
            rename_keys.update(**{value: key})
            data = dataset[value].values
            # Add dimension to specific values that need to be 2D
            """if value not in dataset.coords and len(data.shape) == 1 and data.shape[0] == n_data:
                data_array = dataset[value]
                data_array = data_array.expand_dims({"tmp": 1})
                data_array = data_array.transpose("data", ...)
                dataset = dataset.assign(**{value: data_array})

                # print("state data:", data.shape)
                data = data.reshape(-1, 1)"""

            dataset_arrays.update(**{key: data})
            # print(key, ":=", value, "-->", repr(dataset[value]))
        property_keys.update(**{key: key})

    dataset = dataset.rename_vars(rename_keys)

    return n_data, property_keys, dataset_arrays, dataset


def import_shnitsel_static(
    data_path: str,
    prop_keys,
) -> None | Tuple[int, Dict, Dict, xr.Dataset]:
    atom_position_key = prop_keys[property_names.atomic_position]
    E_key = prop_keys[property_names.energy]
    F_key = prop_keys[property_names.force]
    atom_type_key = prop_keys[property_names.atomic_type]
    atomic_state_key = prop_keys[property_names.atomic_state]

    final_prop_keys = {}
    final_prop_keys.update(prop_keys)

    # Load dataset
    dataset: xr.Dataset = sh.open_frames(data_path)

    varkeys = list(dataset.variables.keys())
    if (
        atom_position_key not in varkeys
        or E_key not in varkeys
        or F_key not in varkeys
        or atom_type_key not in varkeys
        or atomic_state_key not in varkeys
    ):
        logging.warning(
            f"Trajectory {data_path} is missing one or more of the keys : {atom_position_key}, {atomic_state_key}, {atom_type_key}, {E_key}, {F_key}. It has keys <<{varkeys}>>. The trajectory will be skipped."
        )
        return None

    # Normalize units

    # data already in eV
    # dataset[E_key] = dataset[E_key]
    # convert data to eV from hartree/bohr used in Shnitsel
    print(dataset[F_key].units)
    print(dataset[E_key].units)
    print(dataset[atom_position_key].units)
    dataset[atom_position_key].values *= si.Bohr
    dataset[atom_position_key].assign_attrs(units="angstrom")
    dataset[F_key].values *= si.Hartree / si.Bohr
    dataset[F_key].assign_attrs(units="eV/angstrom")
    dataset[E_key].values *= si.Hartree
    dataset[E_key].assign_attrs(units="eV")

    print(
        "Energy (max/median/avg/min)",
        np.max(dataset[E_key].values),
        np.median(dataset[E_key].values),
        np.average(dataset[E_key].values),
        np.min(dataset[E_key].values),
    )

    forces_norm = np.linalg.norm(dataset[F_key].values, axis=2, ord=2)
    print(
        "Force (max/median/avg/min)",
        np.max(forces_norm),
        np.median(forces_norm),
        np.average(forces_norm),
        np.min(forces_norm),
    )

    symbols = dataset.symbols
    # dataset = dataset.drop_vars("symbols")
    dataset = dataset.assign_coords(atNames=symbols)
    # Fix text labels and change to numbers
    state_size = dataset.coords[atomic_state_key]
    # print("State coord:", repr(state_size.values.shape))
    num_states = state_size.values.shape[0]
    dataset = dataset.assign_coords({"state": np.arange(1, 1 + num_states)})
    # dataset = dataset.rename_vars({"symbols":"atNames"})
    # print(repr(dataset))

    rdkit_mol = dataset.isel(frame=0).positions.sh.to_mol()
    idx_i, idx_j, pair_mask = adjacency_from_mol(rdkit_mol)

    # TODO: use sh.core.geom.identify_bonds() to get bonds or use rdkit directly to construct adjacency matrix.

    # print(repr(dataset.isel(frame=0).atXYZ.sh.to_mol()))
    # print(repr(rdkit_mol))

    # Restructure the dimensions into one continuous frame/state dimension

    lead_dimension = "frame"

    dataset = dataset.transpose(lead_dimension, ...)
    # dataset = dataset.reset_index("frame")

    if lead_dimension in dataset.indexes or lead_dimension in dataset.xindexes:
        dataset = dataset.reset_index(lead_dimension)
    dataset = dataset.stack(data=[lead_dimension, atomic_state_key])
    dataset = dataset.transpose("data", ...)
    lead_dimension = "data"

    # Translate atom types into number representations
    atom_type = dataset.variables[atom_type_key]
    atom_number = transform_atom_name_to_number(atom_type)

    n_data = dataset.sizes[lead_dimension]
    n_atoms = dataset.sizes["atom"]

    # Make atom number array span the entirety of the dataset
    atom_number_array = xr.DataArray(
        atom_number).expand_dims({lead_dimension: n_data})
    atom_number_array = atom_number_array.transpose(lead_dimension, ...)

    # Create adjacency matrix
    atom_idx_i = idx_i.expand_dims({lead_dimension: n_data})
    atom_idx_i = atom_idx_i.transpose(lead_dimension, ...)
    atom_idx_j = idx_j.expand_dims({lead_dimension: n_data})
    atom_idx_j = atom_idx_j.transpose(lead_dimension, ...)
    atom_pair_mask = pair_mask.expand_dims({lead_dimension: n_data})
    atom_pair_mask = atom_pair_mask.transpose(lead_dimension, ...)
    # Create node masks
    node_mask = xr.DataArray(
        np.full((n_atoms,), True, dtype=bool), dims=("atom",), name="nodes_mask"
    ).expand_dims({lead_dimension: n_data})
    node_mask = node_mask.transpose(lead_dimension, ...)

    # Append new variables to dataset
    dataset = dataset.assign(atomic_type=atom_number_array)
    dataset = dataset.assign(idx_i=atom_idx_i)
    dataset = dataset.assign(idx_j=atom_idx_j)
    dataset = dataset.assign(pair_mask=atom_pair_mask)
    dataset = dataset.assign(node_mask=node_mask)
    final_prop_keys[property_names.atomic_type] = "atomic_type"

    # Make state into a per-atom variable
    molecule_state_array = dataset[final_prop_keys[property_names.atomic_state]]
    per_atom_state_array = molecule_state_array.expand_dims({"atom": n_atoms})
    per_atom_state_array = per_atom_state_array.transpose(lead_dimension, ...)
    dataset = dataset.assign(atomic_state_tmp=per_atom_state_array)
    final_prop_keys[property_names.atomic_state] = "atomic_state_tmp"

    # print(repr(dataset))

    # print(repr(dataset["state"]))

    property_keys = dict()
    dataset_arrays = dict()

    rename_keys = dict()

    for key, value in final_prop_keys.items():
        if value in dataset.variables:
            rename_keys.update(**{value: key})
            data = dataset[value].values
            # Add dimension to specific values that need to be 2D
            """if value not in dataset.coords and len(data.shape) == 1 and data.shape[0] == n_data:
                data_array = dataset[value]
                data_array = data_array.expand_dims({"tmp": 1})
                data_array = data_array.transpose("data", ...)
                dataset = dataset.assign(**{value: data_array})

                # print("state data:", data.shape)
                data = data.reshape(-1, 1)"""

            dataset_arrays.update(**{key: data})
            # print(key, ":=", value, "-->", repr(dataset[value]))
        property_keys.update(**{key: key})

    dataset = dataset.rename_vars(rename_keys)

    return n_data, property_keys, dataset_arrays, dataset


if __name__ == "__main__":

    # logging.basicConfig()
    # logging.getLogger().setLevel(logging.DEBUG)

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run a training run on the shnitsel dataset."
    )

    # Add the arguments
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=False,
        default=os.getcwd(),
        help="Path to the checkpoint directory. Defaults to the current directory.",
    )

    parser.add_argument(
        "--continue_from",
        type=str,
        required=False,
        default=None,
        help="Path to a checkpoint directory from which to load the overall model state.",
    )

    """parser.add_argument('--apply_to', type=str, required=False, default=None,
                    help='Path to data file that the model should be applied to. '
                            'Defaults to the training data file.')

    parser.add_argument('--on', type=str, required=False, default='test',
                    help='Evaluate the model on the `train`,`valid` or `test` split. Defaults to `test`.')"""

    # Arguments that determine the training parameters
    parser.add_argument(
        "--n_test",
        type=int,
        required=False,
        default=None,
        help="Number of test points. Defaults to all data points that have been not seen during "
        "training if model is evaluated on the same data set as it has been trained on.",
    )

    parser.add_argument(
        "--n_train",
        type=int,
        required=False,
        default=None,
        help="Number of training points. Defaults to 3/5 of all data points.",
    )

    parser.add_argument(
        "--n_valid",
        type=int,
        required=False,
        default=None,
        help="Number of validation points. Defaults to 1/5 of all data points.",
    )

    parser.add_argument(
        "--n_epochs",
        type=int,
        required=False,
        default=1000,
        help="Number of training epochs. Default=1000.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=5,
        help="Batch size of the inference passes. Default=5",
    )

    parser.add_argument(
        "--degree",
        type=int,
        required=False,
        default=2,
        help="Degree of spherical harmonics considered. Default=2",
    )

    parser.add_argument(
        "--features",
        type=int,
        required=False,
        default=32,
        help="Number of features. Default=32",
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        required=False,
        default=2,
        help="Number of layers. Default=2",
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        required=False,
        default=2,
        help="Number of attention heads. May be corrected down to avoid division by zero. Default=2",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        default=1e-3,
        help="Initial learning rate for exponential decay. Default: 1e-3",
    )

    parser.add_argument(
        "--transition_steps",
        type=int,
        required=False,
        default=10_000,
        help="Number of steps between decay. Default: 10,000",
    )

    parser.add_argument(
        "--decay_factor",
        type=float,
        required=False,
        default=0.99,
        help="Decay factor upon each @p transition_steps steps. Default: 0.99",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        required=False,
        default=None,
        help="Number of steps for initial warmup. Default: None. If set will cause initial warmup",
    )

    parser.add_argument(
        "--warmup_init_value",
        type=float,
        required=False,
        default=None,
        help="Learning rate multiplier at start of warmup. Default: None. Will default to a factor of 0.01 times (multiplied with the learning rate specified)",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        required=False,
        default=None,
        help="Random seed for initialization and other random operations. Default: generated from time",
    )

    # parser.add_argument('--from_split', type=str, required=False, default=None,
    #                  help='The name of the data split. If not specified, all data from the file specified in '
    #                        '`--apply_to` is loaded and used for testing.')

    parser.add_argument(
        "--units",
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1,KEY2=VAL2...",
        default=None,
        help="Units in the data set for the quantities. Needs only to be specified"
        "if the model has been trained on units different from the ones present in the data set.",
    )

    parser.add_argument(
        "--prop_keys",
        action=StoreDictKeyPair,
        metavar="KEY1=VAL1,KEY2=VAL2...",
        default=None,
        help="Property keys of the data set. Needs only to be specified, if e.g. the keys of the "
        "properties in the data set that the model is applied to differ from the keys the model"
        "has been trained on.",
    )

    parser.add_argument(
        "--neigh_cut",
        type=float,
        required=False,
        default=None,
        help="Cutoff used for the calculation of the neighborhood lists. Defaults to the r_cut"
        "of the NN model.",
    )

    parser.add_argument(
        "--ground_state",
        action="store_true",
        help="Only use ground-state data from trajectories.",
    )

    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static trajectory data.",
    )

    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic trajectory data.",
    )

    parser.add_argument("--targets", nargs="+", required=False, default=None)

    """parser.add_argument(
        "--jax_dtype",
        type=str,
        required=False,
        default="x32",
        help="Set JAX default dtype. Default is jax.numpy.float32",
    )"""

    """parser.add_argument(
        "--save_predictions_to",
        type=str,
        required=False,
        default="predictions.npz",
        help="Save the predictions and ground truth values to a ckpt_dir/$save_predictions_to.npz.",
    )"""

    args = parser.parse_args()

    # Read arguments
    ckpt_dir = args.ckpt_dir
    batch_size = args.batch_size
    n_test = args.n_test
    n_train = args.n_train
    n_valid = args.n_valid
    # apply_to = args.apply_to
    # from_split = args.from_split
    units = args.units
    a_prop_keys = args.prop_keys
    n_cut = args.neigh_cut
    # _targets = args.targets
    # save_predictions_to = args.save_predictions_to
    sphc_degree = args.degree
    num_features = args.features
    n_layers = args.n_layers
    n_heads = args.n_heads
    n_epochs = args.n_epochs

    continue_from_path = pathlib.Path(args.continue_from).resolve(
    ).as_posix() if args.continue_from is not None else None
    is_loaded_initial_state = continue_from_path is not None

    p_learning_rate = args.learning_rate
    p_exonential_decay_transition_steps = args.transition_steps
    p_exonential_decay_factor = args.decay_factor

    p_warmup_steps = args.warmup_steps
    p_warmup_init_value = args.warmup_init_value if args.warmup_init_value is not None else 0.01
    has_warmup_options = p_warmup_steps is not None

    p_random_seed = args.random_seed

    p_random_seed = p_random_seed if p_random_seed is not None else int.from_bytes(
        os.urandom(4))
    max_seed_val = 2**32-1
    if p_random_seed > max_seed_val:
        p_random_seed = p_random_seed % max_seed_val
    elif p_random_seed < 0:
        p_random_seed = max_seed_val + (p_random_seed % max_seed_val)

    use_only_ground_state = args.ground_state

    use_static = args.static
    use_dynamic = args.dynamic

    if not use_static and not use_dynamic:
        logging.error(
            "Either static or dynamic data loading must be enabled if no input archive is denoted."
        )
        sys.exit(1)

    n_heads = max(1, min(num_features, n_heads))
    for candidate in range(n_heads, 0, -1):
        if num_features % candidate == 0:
            n_heads = candidate
            break

    print(f"Opted for {n_heads} attention heads")

    sphc_degrees_array = list(range(1, sphc_degree + 1))

    port = portpicker.pick_unused_port()
    jax.distributed.initialize(
        f"localhost:{port}", num_processes=1, process_id=0)

    data_dynamic_path = "shnitsel_data/I01_43365/I01_ch2nh2_0p50fs_dynamic.nc"
    data_static_path = "shnitsel_data/I01_43365/I01_ch2nh2_static.nc"
    save_path = "ckpt_dir"

    ckpt_dir = (
        pathlib.Path(args.ckpt_dir)
        .joinpath(save_path)
        .joinpath(f"shnitsel_L{n_layers}_F{num_features}_deg{sphc_degree}")
        .absolute()
        .resolve()
    ).as_posix()

    ckpt_dir = create_directory(ckpt_dir, exists_ok=False)

    if a_prop_keys is not None:
        prop_keys_shnitsel_dynamic.update(a_prop_keys)

    data_path = "./shnitsel_data/"

    shnitsel_base_path = pathlib.Path("./shnitsel_data/")

    system_input = get_full_system_list(shnitsel_base_path)

    # print(repr(system_input))
    n_data_total, loaded_systems = load_system_data(
        system_input, use_dynamic_traj=use_dynamic, use_static_traj=use_static
    )

    print(n_data_total)

    n_data, prop_keys_final, dataset_arrays = merge_system_data_set(
        loaded_systems)

    if use_only_ground_state:
        filtered_dict = {}
        state_filter = np.any(
            dataset_arrays[prop_keys_final[pn.atomic_state]] == 1, axis=1
        )
        new_n_data = np.sum(state_filter)
        for k, v in dataset_arrays.items():
            if v.shape[0] == n_data:
                v = v[state_filter, ...]
                filtered_dict.update({k: v})
            else:
                filtered_dict.update({k: v})
        dataset_arrays = filtered_dict

        print(
            f"Filtered out {new_n_data} ground states of {n_data} frames for training"
        )
        n_data = new_n_data

    dataset_arrays[pn.idx_i][np.invert(dataset_arrays[pn.pair_mask])] = -1
    dataset_arrays[pn.idx_j][np.invert(dataset_arrays[pn.pair_mask])] = -1

    print(n_data)
    # print(repr(loaded_systems))
    # sys.exit(1)

    """n_data, prop_keys_final, dataset_arrays, dataset = import_shnitsel_static(
        data_path=data_static_path, prop_keys=prop_keys_shnitsel_static
    )

    n_data, prop_keys_final, dataset_arrays, dataset = import_shnitsel_dynamic(
        data_path=data_dynamic_path, prop_keys=prop_keys_shnitsel_dynamic
    )"""

    prop_keys = prop_keys_final

    num_training = n_train if n_train is not None else int(
        np.round(n_data * 0.6))
    num_test = n_test if n_test is not None else int(np.round(n_data * 0.2))
    num_valid = n_valid if n_valid is not None else int(np.round(n_data * 0.2))
    num_valid = n_data - num_test - num_training

    r_cut = n_cut if n_cut is not None else 5
    data_set = DataSet(data=dataset_arrays, prop_keys=prop_keys)
    data_set.random_split(
        n_train=num_training,
        n_valid=num_valid,
        n_test=None,  # num_test,
        mic=False,
        r_cut=r_cut,
        training=True,
        seed=p_random_seed,
    )

    data_set.save_splits_to_file(ckpt_dir, "splits.json")
    opt = Optimizer()
    tx = opt.get(learning_rate=p_learning_rate)
    d = data_set.get_data_split()

    if not is_loaded_initial_state:
        # TODO: FIXME: Check whether this shift is actually reasonable
        # TODO: FIXME: I believe that the per-atom contribution is not actually useful because it does not account for the node mask
        data_set.shift_x_by_mean_x(x=pn.energy)
        data_set.save_scales(ckpt_dir, "scales.json")

        net = So3krates(
            F=num_features,
            n_layer=n_layers,
            prop_keys=prop_keys,
            # Add state embedding
            embeddings=[
                AtomTypeEmbed(
                    num_embeddings=100, features=num_features, prop_keys=prop_keys
                ),
                MolecularStateEmbed(
                    num_embeddings=20, features=num_features, prop_keys=prop_keys
                ),
            ],
            geometry_embed_kwargs={
                "degrees": sphc_degrees_array, "r_cut": r_cut},
            so3krates_layer_kwargs={"n_heads": n_heads,
                                    "degrees": sphc_degrees_array},
        )

        coach = Coach(
            # inputs=[pn.atomic_position, pn.atomic_type, pn.atomic_state, pn.idx_i, pn.idx_j, pn.node_mask],
            inputs=[
                pn.atomic_position,
                pn.atomic_type,
                pn.atomic_state,
                pn.idx_i,
                pn.idx_j,
                pn.node_mask,
            ],
            targets=[pn.energy, pn.force],
            epochs=n_epochs,
            training_batch_size=batch_size,
            validation_batch_size=batch_size,
            # TODO: Think about correct relative weight for energy and force
            # loss_weights={pn.energy: 1./8., pn.force: 2e3},
            loss_weights={pn.energy: 0.01, pn.force: 0.99},
            ckpt_dir=ckpt_dir,
            data_path=data_path,
            net_seed=p_random_seed,
            training_seed=p_random_seed,
        )

        data_tuple = DataTuple(
            inputs=coach.inputs, targets=coach.targets, prop_keys=prop_keys
        )

        train_ds = data_tuple(d["train"])

        inputs = jax.tree_util.tree_map(
            lambda x: jnp.array(x[0, ...]), train_ds[0])
        params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
    else:
        logging.info(f"Continue from state at {continue_from_path}")
        net = None
        params = None
        coach = None

        h = read_json(os.path.join(continue_from_path, "hyperparameters.json"))

        # Loading coach configuration from checkpoint to modify for new parameters
        old_coach = Coach(**h["coach"])

        targets = old_coach.targets

        # updated configurable parameters of coach
        old_coach.epochs = n_epochs
        old_coach.training_batch_size = batch_size
        old_coach.validation_batch_size = batch_size
        old_coach.ckpt_dir = ckpt_dir
        old_coach.data_path = data_path
        old_coach.net_seed = p_random_seed
        old_coach.training_seed = p_random_seed

        coach = old_coach

        # Initializing the net from stored hyper parameters
        old_net = init_stack_net(h)
        _prop_keys = old_net.prop_keys
        if prop_keys is not None:
            _prop_keys.update(prop_keys)
            old_net.reset_prop_keys(prop_keys=_prop_keys)
        prop_keys = old_net.prop_keys

        net = old_net

        data_tuple = DataTuple(
            inputs=coach.inputs, targets=coach.targets, prop_keys=prop_keys
        )

        train_ds = data_tuple(d["train"])

        # Extract cut parameters of the model from checkpoint
        r_cut = [
            x[list(x.keys())[0]]["r_cut"]
            for x in h["stack_net"]["geometry_embeddings"]
            if list(x.keys())[0] == "geometry_embed"
        ][0]

        mic = [
            x[list(x.keys())[0]]["mic"]
            for x in h["stack_net"]["geometry_embeddings"]
            if list(x.keys())[0] == "geometry_embed"
        ][0]

        if n_cut is not None:
            r_cut = n_cut
            if n_cut < r_cut:
                logging.warning(
                    f"The specified cutoff for neighborhood calculations n_cut={n_cut} is smaller than the "
                    f"model cutoff r_cut={r_cut}. This will likely result in wrong model prediction."
                )

        # Load old net parameters for initialization
        old_params = load_params_from_ckpt_dir(continue_from_path)
        params = old_params

        # Deal with scaling applied in original network
        scales = read_json(os.path.join(continue_from_path, "scales.json"))

        # TODO: FIXME: Deal with the weird offset and scaling system of So3krates
        p_e_offset = scales[pn.energy]["per_atom_shift"][1]

        # Apply old shift to new dataset and store to ckpt directory
        data_set.shift_x_by_offset(x=pn.energy, offset=p_e_offset)
        data_set.save_scales(ckpt_dir, "scales.json")

    valid_ds = data_tuple(d["valid"])

    obs_fn = get_obs_and_force_fn(net)
    obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

    loss_fn = get_loss_fn(
        obs_fn=obs_fn, weights=coach.loss_weights, prop_keys=prop_keys
    )

    train_state, h_train_state = create_train_state(
        net,
        params,
        tx,
        polyak_step_size=None,
        plateau_lr_decay={"patience": 50, "decay_factor": 1.0},
        scheduled_lr_decay={
            "exponential": {"transition_steps": p_exonential_decay_transition_steps, "decay_factor": p_exonential_decay_factor}
        },
        lr_warmup={
            'init_value': p_warmup_init_value,
            'peak_value': 1,
            'warmup_steps': p_warmup_steps,
        } if has_warmup_options else None,
    )

    h_net = net.__dict_repr__()
    h_opt = opt.__dict_repr__()
    h_coach = coach.__dict_repr__()
    h_dataset = data_set.__dict_repr__()
    h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
    save_dict(path=ckpt_dir, filename="hyperparameters.json",
              data=h, exists_ok=True)

    wandb.init(config=h)
    coach.run(
        train_state=train_state,
        train_ds=train_ds,
        valid_ds=valid_ds,
        loss_fn=loss_fn,
        log_every_t=1,
        restart_by_nan=True,
        use_wandb=True,
    )
