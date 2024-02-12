from plot_traj_with_brt import *
from scipy.io import loadmat
import pickle

brt_tensor_dict = loadmat( "brt_dubin_new_map_disturbed_aug_16_highres.mat", simplify_cells=True )
start_goal_data = pickle.load(open( "state_pairs_outside_disturbed_brt_dec_29.pkl", "rb" ) )

config_names  = ['Vanilla MPPI', 'Vanilla with BRT cost', 'filtered obs cost', 'OUR METHOD', 'Vanilla with samples at limits' ]



trial = 55
config = 0

expr_data = pickle.load( open( f"results/results_trial_{trial}_config_{config}.pkl", "rb" ) )


title  = f"Trial {trial}, {config_names[config]}"
map_data  = pickle.load( open( "new_map_aug_12.pkl", "rb" ) )


plot_trajectory_with_brt_NEW(map_data, expr_data, brt_tensor_dict, start_goal_data,
                             show_brt = False,    # Try setting this to True to visualize BRT (makes plot object much larger...)
                             title = title,
                             trial = trial,
                             frames_to_plot = None,  # This will plot all the frames. Truncate with e.g. range(120, 173) 
                             max_samples = 200)
