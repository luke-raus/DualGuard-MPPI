import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})


exp_summaries_fname = Path('experiments_nov_6_no_lookahead') / '_stats' / 'exp_summaries.csv'

df = pd.read_csv(exp_summaries_fname)

controllers    = df['control_profile'].unique().tolist()
samples_values = [1000, 250, 20] #[1000, 250, 60, 20]  #df['mppi_samples'].unique().tolist()

n_controllers = 6


# Get the data!!!
data = {}
for (num_samples, controller), group in df.groupby(['mppi_samples', 'control_profile']):

    num_trials = len(group)
    num_crashed = int( group['crashed'].sum() )
    num_timed_out = len( group[(group['crashed'] == False) & (group['goal_reached'] == False)] )

    finished_ep_costs = group['total_cost'].loc[group['goal_reached'] == True].to_list()

    data[(controller, num_samples)] = (num_crashed, num_timed_out, finished_ep_costs)

if n_controllers == 4:
    controller_arrangement = [
        ['Vanilla MPPI with obstacle costs',  'Filtered MPPI with obstacle costs'],
        ['Shield MPPI',                       'Sample-safe MPPI (our method)']
    ]
elif n_controllers == 6:
    controller_arrangement = [
        ['Vanilla MPPI with obstacle costs',  'Vanilla MPPI with BRT costs'],
        ['Filtered MPPI with obstacle costs', 'Filtered MPPI with BRT costs'],
        ['Shield MPPI',                       'Sample-safe MPPI (our method)']
    ]

controller_name_conversion = {
    'Vanilla MPPI with obstacle costs':  'Obstacle penalty',
    'Vanilla MPPI with BRT costs':       'BRT penalty',
    'Filtered MPPI with obstacle costs': 'Obstacle pen. + LRF',
    'Filtered MPPI with BRT costs':      'BRT pen. + LRF',
    'Shield MPPI':                       'Shield-MPPI',
    'Sample-safe MPPI (our method)':     'DualGuard MPPI (Ours)',
}


def create_bar_data(num, max, bin_range:tuple, bin_w):
    ngroups   = num // max
    remainder = num % max
    bin_edges = range(bin_range[0], bin_range[1], bin_w)
    bin_h = [max]*ngroups + [remainder] + [0]*(len(bin_edges)-ngroups-1)
    return bin_edges, bin_h


nrows, ncols = len(controller_arrangement), len(controller_arrangement[0])

# fig = plt.figure(figsize=(5, nrows*1.6))   # (5, 3.2) for 2x2 grid; 1.6 vertical units per row
fig = plt.figure(figsize=(5, nrows*1.33))   # (5, 3.2) for 2x2 grid; 1.6 vertical units per row
subfigs = fig.subfigures(nrows, ncols, wspace=0.04, hspace=0.11) #0.07

for row in range(nrows):
    for col in range(ncols):

        subfig = subfigs[row, col]

        controller = controller_arrangement[row][col]
        short_controller_name = controller_name_conversion[controller]
        subfig.suptitle(f'{short_controller_name}', weight='bold', y=1.01)

        axs = subfig.subplots(len(samples_values), 1)#, gridspec_kw={'hspace':0.2})

        # dist_ax.set_xlabel('Episode cost distribution')

        for samp_ind, num_samples in enumerate(samples_values):

            # ---- Retrieve data ----
            num_crashed, num_timed_out, finished_ep_costs = data[(controller, num_samples)]

            num_succeed = len(finished_ep_costs)

            num_trials = num_succeed + num_timed_out + num_crashed
            assert num_trials == 100

            # ---- Cost distribution plot ----

            dist_ax = axs[samp_ind]

            spine_tick_width = 0.6

            # Costs on X-axis
            x_min = 0
            x_dist_max = 30000
            x_max = 43000
            bin_w = 1000
            x_tick_interval = 10000

            timed_out_ax_lims = (32000, 36000)
            crashed_ax_lims   = (37000, 42000)

            xticks_unlabeled =  [10000, 20000, 30000, 34000, 39500] #range(x_min, x_dist_max+1, x_tick_interval)
            xticks_labeled = [0, 10000, 20000, 30000, 34000, 39500]

            xtick_labels  = ['0', '10k ', '20k ', '30k+   ', 'T', 'C']    # '≥30k'

            cost_bins = range(x_min, x_dist_max+1, bin_w)
            # Clip any observations above x_dist_max into the "x_dist_max & up" bin; https://stackoverflow.com/a/30305331
            hist_data = np.clip(finished_ep_costs, cost_bins[0], cost_bins[-1])

            hist = np.histogram(hist_data, bins=cost_bins)
            dist_ax.bar(hist[1][:-1], hist[0], width=bin_w, align='edge', bottom=0.3, color='tab:blue', zorder=3)

            # Plot histogram
            # dist_ax.hist(hist_data, bins=cost_bins, color='tab:blue', label='Cost Distribution')

            dist_ax.set_xlim(x_min, x_max)
            dist_ax.set_xticks(xticks_unlabeled, labels=[])
            dist_ax.tick_params(axis='x', which='major', labelsize=8, length=2, width=spine_tick_width)

            # Y-axis limits & ticks
            y_min = 0
            y_max = 20

            yticks       = [0, 10, 20]
            ytick_labels = ['', '10', '20']
            if samp_ind == len(samples_values)-1:
                ytick_labels[0] = '0'

            dist_ax.set_ylim(y_min, y_max)
            dist_ax.set_yticks(yticks, labels=ytick_labels)
            dist_ax.tick_params(axis='y', which='major', labelsize=5, length=2, width=spine_tick_width)

            dist_ax.spines.left.set_bounds(y_min, y_max)
            dist_ax.spines.bottom.set_bounds(x_min, x_dist_max)

            # Remove top & right spines
            dist_ax.spines.top.set_visible(False)
            dist_ax.spines.right.set_visible(False)
            dist_ax.spines.left.set_linewidth(spine_tick_width)
            dist_ax.spines.bottom.set_linewidth(spine_tick_width)
            dist_ax.set_axisbelow(True)

            other_ax = dist_ax.twinx()
            other_ax.spines.top.set_visible(False)
            other_ax.spines.left.set_visible(False)
            other_ax.spines.right.set_visible(False)
            other_ax.spines.bottom.set_bounds(timed_out_ax_lims[0], timed_out_ax_lims[1])
            other_ax.spines.bottom.set_linewidth(spine_tick_width)
            other_ax.set_yticks([])

            other_ax2 = dist_ax.twinx()
            other_ax2.spines.top.set_visible(False)
            other_ax2.spines.left.set_visible(False)
            other_ax2.spines.right.set_visible(False)
            other_ax2.spines.bottom.set_bounds(crashed_ax_lims[0], crashed_ax_lims[1])
            other_ax2.spines.bottom.set_linewidth(spine_tick_width)
            other_ax2.set_yticks([])

            # Create bar graph for timed out trials
            bin_edges, bin_h = create_bar_data(num_timed_out, y_max, timed_out_ax_lims, bin_w)
            dist_ax.bar(bin_edges, bin_h, width=bin_w, bottom=spine_tick_width/2, align='edge', color='tab:orange')

            # Create bar graph for crashed trials
            bin_edges, bin_h = create_bar_data(num_crashed, y_max, crashed_ax_lims, bin_w)
            dist_ax.bar(bin_edges, bin_h, width=bin_w, bottom=spine_tick_width/2, align='edge', color='tab:red')

            # Number of samples label
            dist_ax.text(x=-4700, y=9, s=f'{num_samples}', weight='bold', ha='right', va='center')

            # Set labels only on the last subplot
            if samp_ind == len(samples_values)-1:

                dist_ax.set_xticks(xticks_labeled, labels=xtick_labels)

                # Write "Samples" vertically on leftmost column
                if col == 0:
                    # Since we're rotation, ha & va are flipped
                    dist_ax.text(x=-12300, y=32, s='Samples', rotation=90, ha='center', va='center')
                    # y=44

                # Add x-axis label to bottom row
                if row == nrows-1:
                    dist_ax.set_xlabel('Episode cost or outcome')


plt.savefig(f'experiments_nov_6_no_lookahead/_figures/dubins_sim_cost_dists_{n_controllers}_shorter.pdf',
            format='pdf', bbox_inches='tight', pad_inches=0.02)
