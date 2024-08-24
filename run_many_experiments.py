from run_experiments_omegaconf_flat import run_experiment

from omegaconf import OmegaConf
from pathlib import Path
import itertools


num_samples_opts    = [20, 36, 50, 100, 250, 500, 1000, 2000, 5000]
filter_samples_opts = [True, False]
cost_type_opts      = ['obs', 'brt']

default_config_fname = 'default_config.yaml'
experiments_path = Path('experiments')

experiment_id = 0


# === Setup experiment configurations ===

# Iterate thru every combination of parameter options
param_combos = itertools.product(num_samples_opts, filter_samples_opts, cost_type_opts)

for param_combo in param_combos:

    num_samples, filter_samples, cost_type = param_combo

    # Load default config
    config = OmegaConf.load(default_config_fname)

    # Override defaults with this experiment's settings
    config.apply_safety_filter_to_samples = filter_samples
    config.cost_from_obstacles_or_BRT = cost_type
    config.mppi_samples = num_samples

    # Set & create experiment directory, then save config file
    experiment_dir = experiments_path / f'experiment_{experiment_id:04}'
    experiment_dir.mkdir(parents=True, exist_ok=False)
    # May want to let user know that we're avoiding overriding existing experiment dirs
    OmegaConf.save(config, f=experiment_dir/'config.yaml')

    experiment_id += 1


# === Run experiments ===

for experiment_dir in sorted(experiments_path.iterdir()):

    print(f'Running experiment {experiment_dir}')
    run_experiment(experiment_dir/'config.yaml', experiment_dir)

print('All done!')

# config_name = f'trial-{trial_num}_samples-{}filter-{filter_samples}_cost-{cost_type}'
