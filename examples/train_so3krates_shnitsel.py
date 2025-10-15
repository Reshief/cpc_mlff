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
from mlff.properties import shnitsel_property_keys_dynamic as prop_keys
from mlff.properties import property_names

import mlff.properties.property_names as pn
from netCDF4 import Dataset
import sys

import shnitsel as sh
import xarray as xr
from shnitsel.core.parse.common import transform_atom_name_to_number

import argparse

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

n_heads = max(1, min(num_features, n_heads))
print(f"Opted for {n_heads} attention heads")

sphc_degrees_array = list(range(1, sphc_degree + 1))


port = portpicker.pick_unused_port()
jax.distributed.initialize(f"localhost:{port}", num_processes=1, process_id=0)

data_dynamic_path = "data/I01_43365/I01_ch2nh2_0p50fs_dynamic.nc"
data_static_path = "data/I01_43365/I01_ch2nh2_static.nc"
save_path = "ckpt_dir"

import pathlib

ckpt_dir = (
    pathlib.Path(args.ckpt_dir)
    .joinpath(f"module_F{num_features}_deg{sphc_degree}")
    .absolute()
    .resolve()
).as_posix()
ckpt_dir = create_directory(ckpt_dir, exists_ok=False)

if a_prop_keys is not None:
    prop_keys.update(a_prop_keys)

E_key = prop_keys[property_names.energy]
F_key = prop_keys[property_names.force]
atom_type_key = prop_keys[property_names.atomic_type]

data_path = data_static_path
data_path = data_dynamic_path

dataset: xr.Dataset = sh.open_frames(data_path)
dataset = dataset.transpose("frame", ...)
dataset = dataset.reset_index("frame")
dataset = dataset.stack(data=["frame", "state"])
dataset = dataset.transpose("data", ...)
# print(repr(dataset))

atom_type = dataset.variables[atom_type_key]
atom_number = transform_atom_name_to_number(atom_type)

n_data = dataset.sizes["data"]
n_atoms = dataset.sizes["atom"]
# print(n_data)

positions = dataset.variables[prop_keys[property_names.atomic_position]]

# data already in eV
# dataset[E_key] = dataset[E_key]
# convert data to eV from hartree/bohr used in Shnitsel
dataset[F_key].values *= si.Bohr / si.Hartree
dataset[F_key].assign_attrs(units="eV/m")

atom_number_array = xr.DataArray(atom_number).expand_dims({"data": n_data})
atom_number_array = atom_number_array.transpose("data", ...)

atom_idx_i = xr.DataArray([0, 1], dims=("pair_index",), name="idx_i").expand_dims(
    {"data": n_data}
)
atom_idx_i = atom_idx_i.transpose("data", ...)
atom_idx_j = xr.DataArray([1, 0], dims=("pair_index",), name="idx_j").expand_dims(
    {"data": n_data}
)
atom_idx_j = atom_idx_j.transpose("data", ...)
node_mask = xr.DataArray(
    np.full((n_atoms,), True, dtype=bool), dims=("atom",), name="nodes_mask"
).expand_dims({"data": n_data})
node_mask = node_mask.transpose("data", ...)

dataset = dataset.assign(atomic_type=atom_number_array)
dataset = dataset.assign(idx_i=atom_idx_i)
dataset = dataset.assign(idx_j=atom_idx_j)
dataset = dataset.assign(node_mask=node_mask)
prop_keys[property_names.atomic_type] = "atomic_type"
# TODO: FIXME: Deal with the state being one of the features
# dataset = dataset.isel(state=0)

print(repr(dataset))

# print(repr(dataset["state"]))


property_keys = dict()
dataset_arrays = dict()

for key, value in prop_keys.items():
    if value in dataset.variables:
        data = dataset[value].values
        if key == "atomic_state":
            # print("state data:", data.shape)
            data = data.reshape(-1, 1)
            # print("state data:", data.shape)
            dataset_arrays.update(**{key: data})
        else:
            dataset_arrays.update(**{key: data})
        # print(key, ":=", value, "-->", repr(dataset[value]))
    property_keys.update(**{key: key})

prop_keys = property_keys

num_training = int(np.round(n_data * 0.2))
num_valid = int(np.round(n_data * 0.6))

r_cut = n_cut if n_cut is not None else 5
data_set = DataSet(data=dataset_arrays, prop_keys=prop_keys)
data_set.random_split(
    n_train=num_training,
    n_valid=num_valid,
    n_test=None,
    mic=False,
    r_cut=r_cut,
    training=True,
    seed=0,
)

data_set.shift_x_by_mean_x(x=pn.energy)

data_set.save_splits_to_file(ckpt_dir, "splits.json")
data_set.save_scales(ckpt_dir, "scales.json")

d = data_set.get_data_split()

net = So3krates(
    F=num_features,
    n_layer=n_layers,
    prop_keys=prop_keys,
    geometry_embed_kwargs={"degrees": sphc_degrees_array, "r_cut": r_cut},
    so3krates_layer_kwargs={"n_heads": n_heads, "degrees": sphc_degrees_array},
)

obs_fn = get_obs_and_force_fn(net)
obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

opt = Optimizer()

tx = opt.get(learning_rate=1e-3)

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
    epochs=1000,
    training_batch_size=batch_size,
    validation_batch_size=batch_size,
    loss_weights={pn.energy: 0.01, pn.force: 0.99},
    ckpt_dir=ckpt_dir,
    data_path=data_path,
    net_seed=0,
    training_seed=0,
)

loss_fn = get_loss_fn(obs_fn=obs_fn, weights=coach.loss_weights, prop_keys=prop_keys)

data_tuple = DataTuple(inputs=coach.inputs, targets=coach.targets, prop_keys=prop_keys)

train_ds = data_tuple(d["train"])
valid_ds = data_tuple(d["valid"])

inputs = jax.tree_util.tree_map(lambda x: jnp.array(x[0, ...]), train_ds[0])
params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)
train_state, h_train_state = create_train_state(
    net,
    params,
    tx,
    polyak_step_size=None,
    plateau_lr_decay={"patience": 50, "decay_factor": 1.0},
    scheduled_lr_decay={
        "exponential": {"transition_steps": 10_000, "decay_factor": 0.9}
    },
)

h_net = net.__dict_repr__()
h_opt = opt.__dict_repr__()
h_coach = coach.__dict_repr__()
h_dataset = data_set.__dict_repr__()
h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
save_dict(path=ckpt_dir, filename="hyperparameters.json", data=h, exists_ok=True)

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
