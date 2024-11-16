import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py


environment = 'config/brt_dubins_cluttered_0.hdf5'

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})


def plot_environment():
    fig, ax = plt.subplots(figsize=(3, 3))

    obs_color = '0.3'
    brt_color = '#1D918C'

    theta_ind = 11

    # Plot BRT slice
    with h5py.File(environment, 'r') as hdf_file:
        # obs_value   = hdf_file['obstacle_value_grid'][:]
        brt_value     = hdf_file['brt_value_grid'][:]
        brt_grid_axes = tuple([ hdf_file['brt_axes'][f'axis_{i}'][:] for i in range(brt_value.ndim) ])
    theta = brt_grid_axes[2][theta_ind]
    #print(theta)
    X, Y = np.meshgrid(brt_grid_axes[0], brt_grid_axes[1])
    Z = np.transpose(brt_value[:,:,theta_ind])  # Not xactly sure why I need to do this transpose, but...
    ax.contourf( X, Y, Z, levels=[-100, 0], colors=[brt_color])

    # Plot cirular obstacles
    with h5py.File(environment, 'r') as f:
        x = f['obstacles']['obstacle_x'][:]
        y = f['obstacles']['obstacle_y'][:]
        r = f['obstacles']['obstacle_radius'][:]
    for i in range(len(x)):
        ax.add_patch(patches.Circle((x[i], y[i]), r[i], linewidth=0, facecolor=obs_color))

    # Plot walls
    w   = 5    # wall distances from center
    w_t = 2    # wall display thickness
    ax.add_patch(patches.Rectangle((-(w+w_t), -(w+w_t)), 2*(w+w_t),       w_t, linewidth=0, facecolor=obs_color))  # bottom
    ax.add_patch(patches.Rectangle((-(w+w_t), -(w+w_t)),       w_t, 2*(w+w_t), linewidth=0, facecolor=obs_color))  # left
    ax.add_patch(patches.Rectangle((-(w+w_t),        w), 2*(w+w_t),       w_t, linewidth=0, facecolor=obs_color))  # top
    ax.add_patch(patches.Rectangle((       w, -(w+w_t)),       w_t, 2*(w+w_t), linewidth=0, facecolor=obs_color))  # right

    # Set view & axis details
    view = 5.3
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)

    ax.set_xticks([-w, 0, w])
    ax.set_yticks([-w, 0, w])

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')


    plt.savefig(f'dubins_sim_BRT.pdf', format='pdf', bbox_inches='tight')

plot_environment()