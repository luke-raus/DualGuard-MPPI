from scipy.io import loadmat
import json
import h5py


brt_file = 'config_data/brt_dubin_new_map_disturbed_aug_16_fixed_init_value.mat'

brt_mat = loadmat(brt_file, simplify_cells=True)

obs_value       = brt_mat['init_value']
brt_value       = brt_mat['value']
brt_theta_deriv = brt_mat['theta_deriv']

brt_grid_axes = tuple( brt_mat['grid_axes'] )


json_file = 'config_data/dubin_environment_obstacles.json'
with open(json_file, 'r') as f:
    obs_json = json.load(f)



with h5py.File('brt.hdf5', 'w') as f:
    # Save the arrays to the file
    f.create_dataset('obstacle_value_grid', data = obs_value)
    f.create_dataset('brt_value_grid', data = brt_value)
    f.create_dataset('brt_theta_deriv_grid', data = brt_theta_deriv)

    axes = f.create_group('brt_axes')
    for i, axis in enumerate(brt_grid_axes):
        axes.create_dataset(f'axis_{i}', data = brt_grid_axes[i])
    
    obstacles = f.create_group('obstacles')
    obstacles.create_dataset('obstacle_x', data = obs_json['x'])
    obstacles.create_dataset('obstacle_y', data = obs_json['y'])
    obstacles.create_dataset('obstacle_radius', data = obs_json['r'])

