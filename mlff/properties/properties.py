from .property_names import *

md17_property_keys = {
    energy: 'E',
    force: 'F',
    atomic_position: 'R',
    atomic_type: 'z',
    idx_i: 'idx_i',
    idx_j: 'idx_j',
    unit_cell: 'unit_cell',
    pbc: 'pbc',
    cell_offset: 'cell_offset',
    stress: 'stress',
    node_mask: 'node_mask'}

qm7x_property_keys = {
    energy: 'ePBE0+MBD',
    force: 'totFOR',
    atomic_position: 'atXYZ',
    atomic_type: 'atNUM',
    hirshfeld_volume: 'hVOL',
    hirshfeld_volume_ratio: 'hRAT',
    partial_charge: 'hCHG',
    total_dipole_moment: 'vDIP',
    idx_i: 'idx_i',
    idx_j: 'idx_j',
    node_mask: 'node_mask'
}

shnitsel_property_keys_static = {
    energy: 'energy',
    force: 'forces',
    atomic_position: 'positions',
    atomic_type: 'symbols',
    total_dipole_moment: 'dip_perm',
    total_dipole_moment_transient: 'dip_trans',
}

shnitsel_property_keys_dynamic = {
    energy: 'energy',
    force: 'forces',
    atomic_position: 'atXYZ',
    atomic_type: 'atNames',
    atomic_state: 'state',
    total_dipole_moment: 'dip_perm',
    total_dipole_moment_transient: 'dip_trans',
}

'sdiag', 'astate', 'phases', 'nacs', 'from', 'to', 'state', 'state2', 'atom', 'direction', 'atNames', 'max_ts', 'completed', 'nsteps', 'time', 'trajid'