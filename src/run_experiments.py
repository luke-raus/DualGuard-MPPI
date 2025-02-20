from math import exp
from omegaconf import OmegaConf
from pathlib import Path
from experiment_batch import initialize_experiment_batch, run_experiment_batch
from analyze_experiments import analyze_experiment_batch
import sys


experiment_batch_dir     = Path('experiments')

default_config_fname     = Path('config') / 'default_config.yaml'
controller_configs_fname = Path('config') / 'control_profiles.yaml'
episode_configs_fname    = Path('config') / 'episode_params.yaml'

save_samples = True   # Set to False to save storage space, at the expense of
                      # not being able to visualize full detailed results

if __name__ == '__main__':

    if len(sys.argv) == 2:
        batch_size = sys.argv[1]
    else:
        batch_size = 'minimal'   # Change this to run larger experiment sets described below
        print('Defaulting to minimal batch.')

    episode_configs = OmegaConf.load(episode_configs_fname)

    if batch_size == 'minimal':
        episode_configs = episode_configs[23:24]  # Good demo episode
        num_samples_settings = [1000]
    elif batch_size == 'small':
        episode_configs = episode_configs[23:33]
        num_samples_settings = [60, 1000]
    elif batch_size == 'full':
        episode_configs = episode_configs[0:100]
        num_samples_settings = [20, 60, 250, 1000]
    else:
        raise ValueError('Given batch size is not recognized. Try minimal, small, or full')

    initialize_experiment_batch(
        batch_path = experiment_batch_dir,
        default_config     = OmegaConf.load(default_config_fname),
        controller_configs = OmegaConf.load(controller_configs_fname),
        episode_configs    = episode_configs,
        num_samples_settings = num_samples_settings,
        save_samples = save_samples
    )

    run_experiment_batch(experiment_batch_dir)

    analyze_experiment_batch(experiment_batch_dir)

    print('To visualize these results:')
    print('Run: [python plot_experiment_results.py] to see details (e.g. sampled trajectories) for each controller.')
    print('Run: [python plot_experiment_comparison.py] to compare overall controller trajectories per episode.')
