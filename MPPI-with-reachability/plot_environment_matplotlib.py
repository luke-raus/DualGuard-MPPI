import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.axes as axes
import numpy as np
import h5py
from pathlib import Path

from experiment_storage import ExperimentStorage


environment = 'config/brt_dubins_cluttered_0.hdf5'

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})


def plot_environment(ax=None, brt_slice=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig = ax.figure

    if brt_slice is not None:
        brt_color = '#1D918C'

        theta_ind = brt_slice

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

    obs_color = '0.15'

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

    ax.set_xticks([-w, 0, w])
    ax.set_yticks([-w, 0, w])

    return fig, ax


def plot_trajectory(ax:axes.Axes, result:ExperimentStorage, label, color, linestyle='-'):
    trajectory = result.get_overall_trajectory()
    ax.plot(trajectory[:, 0], trajectory[:,1],
            label=label, color=color, linewidth=3.0, ls=linestyle)


def create_BRT_plot():
    fig, ax = plot_environment(brt_slice=11)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # Set view & axis details
    view = 5.3
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)

    fig.savefig(f'figures/dubins_environment_BRT.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0.02)


def create_traj_comparison_plot(exp_dir:Path, exp_base:str):

    fig, ax = plot_environment(brt_slice=None)

    ax.set_xlabel('X [m]')

    name_map = {
        'Vanilla MPPI with obstacle costs':  ('Obs. penalty',    '#a4cde1'),   # light blue
        'Vanilla MPPI with BRT costs':       ('BRT penalty',     '#1e78b0'),   # blue
        'Filtered MPPI with obstacle costs': ('Obs. pen. + LRF', '#b1dd8e'),   # light green
        'Filtered MPPI with BRT costs':      ('BRT pen. + LRF',  '#369d3b'),   # green
        'Shield MPPI':                       ('Shield-MPPI',     '#694296'),   # purple
        'Sample-safe MPPI (our method)':     ('Our method',      '#ff7e1e')    # orange
    }

    results = [ExperimentStorage(x) for x in sorted(exp_dir.iterdir()) if exp_base in str(x)]

    for res in results:
        label, color = name_map[ res.get_config()['control_profile'] ]

        linestyle = '-'
        if label == 'BRT pen. + LRF':
            linestyle = ':'
        if label == 'Obs. pen. + LRF':
            linestyle = ':'

        plot_trajectory(ax, res, label, color, linestyle)

    init_state = results[0].get_config()['init_state']
    goal_state = results[0].get_config()['goal_state']


    ax.scatter(init_state[0], init_state[1], marker=(3, 0, init_state[2]*180/3.141593 - 90),
            s=100, linestyle='None', label='x_0 (initial)', color='#fffe9e', edgecolors= "black", zorder=3)

    ax.scatter(goal_state[0], goal_state[1],
            s=40, linestyle='None', label='x_g (goal)', color='#24ae5e', edgecolors= "black", zorder=3)

    # Set view & axis details
    view = 5.3
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    fig.savefig(f'figures/dubins_sim_traj_comparison_1.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0.02)


if __name__ == "__main__":
    create_BRT_plot()
    create_traj_comparison_plot(Path('experiments_nov_6_no_lookahead'),
                                'exp_samples-1000_ep-023')
