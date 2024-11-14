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
samples_values = [1000, 250, 60, 20]  #df['mppi_samples'].unique().tolist()



# Get the data!!!
data = {}
for (num_samples, controller), group in df.groupby(['mppi_samples', 'control_profile']):

    num_trials = len(group)
    num_crashed = int( group['crashed'].sum() )
    num_timed_out = len( group[(group['crashed'] == False) & (group['goal_reached'] == False)] )

    finished_ep_costs = group['total_cost'].loc[group['goal_reached'] == True].to_list()

    data[(controller, num_samples)] = (num_crashed, num_timed_out, finished_ep_costs)


# controller_arrangement = [
#     ['Vanilla MPPI with obstacle costs',  'Vanilla MPPI with BRT costs'],
#     ['Filtered MPPI with obstacle costs', 'Filtered MPPI with BRT costs'],
#     ['Shield MPPI',                       'Sample-safe MPPI (our method)']
# ]
controller_arrangement = [
    ['Vanilla MPPI with obstacle costs',  'Filtered MPPI with obstacle costs'],
    ['Shield MPPI',                       'Sample-safe MPPI (our method)']
]


controller_name_conversion = {
    'Vanilla MPPI with obstacle costs':  'Obs cost',
    'Vanilla MPPI with BRT costs':       'BRT cost',
    'Filtered MPPI with obstacle costs': 'Obs cost + LRF',
    'Filtered MPPI with BRT costs':      'BRT cost + LRF',
    'Shield MPPI':                       'Shield MPPI',
    'Sample-safe MPPI (our method)':     'Our method',
}

nrows, ncols = 2, 2

fig = plt.figure(figsize=(5, 3.2))
subfigs = fig.subfigures(nrows, ncols, wspace=0.04, hspace=0.07)

for row in range(nrows):
    for col in range(ncols):

        subfig = subfigs[row, col]

        controller = controller_arrangement[row][col]
        short_controller_name = controller_name_conversion[controller]
        subfig.suptitle(f'{short_controller_name}', weight='bold')

        axs = subfig.subplots(len(samples_values), 1, sharey=True)

        # dist_ax.set_xlabel('Episode cost distribution')

        for samp_ind, num_samples in enumerate(samples_values):

            # ---- Retrieve data ----
            num_crashed, num_timed_out, finished_ep_costs = data[(controller, num_samples)]

            num_succeed = len(finished_ep_costs)

            num_trials = num_succeed + num_timed_out + num_crashed
            assert num_trials == 100


            # ---- Cost distribution plot ----

            dist_ax:plt.Axes = axs[samp_ind]


            # Costs on X-axis
            x_min = 0
            x_max = 30000
            x_bin_interval = 1000
            x_tick_interval = 10000

            xticks = range(x_min, x_max+1, x_tick_interval)
            xtick_labels = [str(x).replace('0000', '0k') for x in xticks]
            xtick_labels[-1] += '+'

            cost_bins = range(x_min, x_max+1, x_bin_interval)

            # Plot histogram
            dist_ax.hist(np.clip(finished_ep_costs, cost_bins[0], cost_bins[-1]),    # https://stackoverflow.com/a/30305331
                    bins=cost_bins, color='tab:blue', label='Cost Distribution')

            # X-axis limits
            dist_ax.set_xlim(x_min, x_max)

            # Y-axis limits & ticks
            dist_ax.set_ylim(0, 19)
            dist_ax.set_yticks([0, 10])
            dist_ax.tick_params(axis='y', which='major', labelsize=5)

            # Remove spines
            dist_ax.spines['top'].set_visible(False)
            dist_ax.spines['right'].set_visible(False)


            # Number of samples label
            if col==0 or col==1:
                dist_ax.text(x=-3600, y=9, s=f'{num_samples}', horizontalalignment='right', verticalalignment='center')

            # # Set labels only on the last subplot
            if samp_ind == len(samples_values) - 1:
                dist_ax.tick_params(left=True, bottom=True)

                dist_ax.set_xticks(xticks, labels=xtick_labels)

                dist_ax.tick_params(axis='x', which='major', labelsize=8)

                if col==0:
                    # Since we're rotation, ha & va are flipped
                    dist_ax.text(x=-9000, y=44, s='Samples', rotation=90, ha='center', va='center')

            else:
                dist_ax.tick_params(left=True, bottom=True)
                dist_ax.set_xticklabels([])


plt.savefig(f'figures_WIP_3/dubins_sim_cost_dists.pdf', format='pdf', bbox_inches='tight')

#plt.show()
