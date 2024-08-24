import h5py
import numpy as np
from scipy.io import loadmat


brt_file = 'config_data/brt_dubin_new_map_disturbed_aug_16_fixed_init_value.mat'

brt_mat = loadmat(brt_file, simplify_cells=True)

obs_value       = brt_mat['init_value']
brt_value       = brt_mat['value']
brt_theta_deriv = brt_mat['theta_deriv']

brt_grid_axes = tuple( brt_mat['grid_axes'] )     # with scipy



with h5py.File('brt.hdf5', 'w') as hdf_file:
    # Save the arrays to the file
    hdf_file.create_dataset('obstacle_grid',  data = obs_value)
    hdf_file.create_dataset('brt_value_grid', data = brt_value)
    hdf_file.create_dataset('brt_theta_deriv_grid', data = brt_theta_deriv)
    hdf_file.create_dataset('brt_axis_0', data = brt_grid_axes[0])
    hdf_file.create_dataset('brt_axis_1', data = brt_grid_axes[1])
    hdf_file.create_dataset('brt_axis_2', data = brt_grid_axes[2])
