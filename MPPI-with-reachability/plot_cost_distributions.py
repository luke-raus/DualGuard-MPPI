import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
})


exp_summaries_fname = Path('experiments_oct_1') / '_stats' / 'exp_summaries.csv'

df = pd.read_csv(exp_summaries_fname)

controllers    = df['control_profile'].unique().tolist()
samples_values = [20, 60, 250, 1000]  #df['mppi_samples'].unique().tolist()



# Get the data!!!
data = {}
for (num_samples, controller), group in df.groupby(['mppi_samples', 'control_profile']):

    num_trials = len(group)
    num_crashed = int( group['crashed'].sum() )
    num_timed_out = len( group[(group['crashed'] == False) & (group['goal_reached'] == False)] )

    finished_ep_costs = group['total_cost'].loc[group['goal_reached'] == True].to_list()

    data[(controller, num_samples)] = (num_crashed, num_timed_out, finished_ep_costs)


controller_name_conversion = {
    'Vanilla MPPI with obstacle costs':  'Obstacle costs',
    'Vanilla MPPI with BRT costs':       'BRT costs',
    'Filtered MPPI with obstacle costs': 'Obstacle costs + LR filter',
    'Filtered MPPI with BRT costs':      'BRT costs + LR filter',
    'Sample-safe MPPI (our method)':     'Proposed method',
}


for i, controller in enumerate(controllers):
    fig, axs = plt.subplots(
        len(samples_values), 2,
        width_ratios=[1, 1.2],   #[1, 2.4]
        figsize=(5.0, 1.4),    #(5.0, 2.1) for for all 7 rows
        gridspec_kw={'hspace': 0.05,    #h(eight)space between rows
                     'wspace': 0.13})   #w(idth)space between cols

    short_controller_name = controller_name_conversion[controller]
    fig.suptitle(f'{short_controller_name}', weight='bold', x=0.02, y=1.04, ha='left')

    for j, num_samples in enumerate(samples_values):

        # ---- Retrieve data ----
        num_crashed, num_timed_out, finished_ep_costs = data[(controller, num_samples)]

        num_succeed = len(finished_ep_costs)

        num_trials = num_succeed + num_timed_out + num_crashed
        assert num_trials == 100


        # ---- Outcome plot ----

        outcome_ax = axs[j, 0]

        outcome_ax.barh(0, num_succeed,                                   color='tab:blue',   label='Reached goal')
        outcome_ax.barh(0, num_timed_out, left=num_succeed,               color='tab:orange', label='Time expired')
        outcome_ax.barh(0, num_crashed,   left=num_succeed+num_timed_out, color='tab:red',    label='Crashed')

        outcome_ax.set_xlim(0, num_trials)

        outcome_ax.set_yticklabels([])

        outcome_ax.spines['top'].set_visible(False)
        outcome_ax.spines['left'].set_visible(False)
        outcome_ax.spines['right'].set_visible(True)

        # Number of samples
        outcome_ax.text(x=-4, y=0, s=f'{num_samples}', horizontalalignment='right', verticalalignment='center')

        # Only put legend on first figure
        if i == 0 and j == 0:
            outcome_ax.legend(bbox_to_anchor=(0.0, 2.3, 2.1, 0.15), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)

        if j == len(samples_values)-1:
            outcome_ax.spines['bottom'].set_visible(True)
            outcome_ax.tick_params(left=False, bottom=True)

            outcome_ax.set_xticks([0, num_trials/2, num_trials])
            outcome_ax.tick_params(axis='x', which='major', labelsize=7)

            # Only put x-label on last figure
            if i == len(controllers)-1:
                outcome_ax.set_xlabel('Outcomes')

            succeed_text = f'{num_succeed} successful'
        else:
            outcome_ax.spines['bottom'].set_visible(False)
            outcome_ax.tick_params(left=False, bottom=False)
            outcome_ax.set_xticklabels([])

            succeed_text = f'{num_succeed}'

        # Number of successful trials
        outcome_ax.text(x=4, y=0, s=succeed_text, horizontalalignment='left', verticalalignment='center',
                        color='white', weight='bold', fontsize=8)

        # ---- Cost distribution plot ----

        dist_ax = axs[j, 1]


        # Costs on X-axis
        x_min = 0
        x_max = 20000
        x_bin_interval = 500
        x_tick_interval = 5000

        xticks = range(x_min, x_max+1, x_tick_interval)
        xtick_labels = [str(x) for x in xticks]
        xtick_labels[-1] += '+'

        cost_bins = range(x_min, x_max+1, x_bin_interval)

        # Counts on Y-axis
        y_mix, y_max = 0, 19


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

        # Set labels only on the last subplot
        if j == len(samples_values) - 1:
            dist_ax.tick_params(left=True, bottom=True)

            dist_ax.set_xticks(xticks)

            dist_ax.set_xticklabels(xtick_labels)
            dist_ax.tick_params(axis='x', which='major', labelsize=8)

            # Only put x-label on last figure
            if i == len(controllers)-1:
                dist_ax.set_xlabel('Episode cost distribution')
        else:
            dist_ax.tick_params(left=True, bottom=False)
            dist_ax.set_xticklabels([])

    plt.savefig(f'figures/{short_controller_name.replace(' ', '_')}_cost_dist.pdf', format='pdf', bbox_inches='tight')

    #plt.show()