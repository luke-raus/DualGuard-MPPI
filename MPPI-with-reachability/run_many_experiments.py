from omegaconf import OmegaConf
from pathlib import Path
import itertools
import json

from experiment import Experiment


save_samples = False

num_samples_opts    = [20, 36, 60, 100, 250, 500, 1000, 2000]
filter_samples_opts = [True, False]
cost_type_opts      = ['obs', 'brt']

with open('config/dubin_environment_state_pairs.json', 'r') as f:
    states = json.load(f)
    init_goal_state_pairs = [ {'init':states['init'][i], 'goal':states['goal'][i]}
                              for i in range(len(states['init'])) ]
    init_goal_state_pairs = init_goal_state_pairs[:5]

default_config_fname = 'config/default_config.yaml'
experiments_path = Path('experiments')


# === Setup experiment configurations ===

# Iterate thru every combination of parameter options
param_combos = itertools.product(num_samples_opts, filter_samples_opts, cost_type_opts, init_goal_state_pairs)

experiment_id = 0
for param_combo in param_combos:

    num_samples, filter_samples, cost_type, init_goal_state_pair = param_combo

    # Load default config
    config = OmegaConf.load(default_config_fname)

    # Override defaults with this experiment's settings
    config.apply_safety_filter_to_samples = filter_samples
    config.cost_from_obstacles_or_BRT = cost_type
    config.mppi_samples = num_samples

    config.init_state = init_goal_state_pair['init']
    config.goal_state = init_goal_state_pair['goal']

    config.save_samples = save_samples

    # Set & create experiment directory, then save config file
    experiment_dir = experiments_path / f'experiment_{experiment_id:05}'
    experiment_dir.mkdir(parents=True, exist_ok=False)
    # May want to let user know that we're avoiding overriding existing experiment dirs
    OmegaConf.save(config, f=experiment_dir/'config.yaml')

    experiment_id += 1


# === Run experiments ===

for experiment_dir in sorted(experiments_path.iterdir()):

    print(f'Running experiment {experiment_dir}')
    experiment = Experiment(experiment_dir)
    experiment.initialize()
    experiment.run_and_save()

print('All done!')

# config_name = f'trial-{trial_num}_samples-{}filter-{filter_samples}_cost-{cost_type}'

