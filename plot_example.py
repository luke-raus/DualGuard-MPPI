from plot_traj_with_brt import *
from scipy.io import loadmat
import pickle
import json

brt_tensor_dict = loadmat( "config_data/brt_dubin_new_map_disturbed_aug_16_fixed_init_value.mat", simplify_cells=True )
start_goal_data = json.load(open("config_data/dubin_environment_state_pairs.json", "r"))

config_names  = ['Vanilla MPPI', 'Vanilla with BRT cost', 'filtered obs cost', 'OUR METHOD', 'Vanilla with samples at limits' ]

trial = 0
config = 4
#expr_fname = f"results/results_trial_{trial}_config_{config}.pkl"
expr_fname = "results/result_with_samples.pkl"

expr_data = pickle.load( open( expr_fname, "rb" ) )

title  = "Experiment"
map_data  = json.load( open( "config_data/dubin_environment_obstacles.json", "r" ) )


plot_trajectory_with_brt_NEW(map_data, expr_data, brt_tensor_dict, start_goal_data,
                             show_brt = False,    # Try setting this to True to visualize BRT (makes plot object much larger...)
                             title = title,
                             trial = trial,
                             frames_to_plot = None,  # This will plot all the frames. Truncate with e.g. range(120, 173) 
                             max_samples = 200)
