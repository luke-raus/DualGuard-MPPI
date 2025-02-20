from omegaconf import OmegaConf
from pathlib import Path
from experiment_batch import initialize_experiment_batch, run_experiment_batch
from analyze_experiments import analyze_experiment_batch


experiment_batch_dir     = Path('experiments')

default_config_fname     = Path('config') / 'default_config.yaml'
controller_configs_fname = Path('config') / 'control_profiles.yaml'
episode_configs_fname    = Path('config') / 'episode_params.yaml'

batch_size = 'minimal'


if __name__ == '__main__':

    if batch_size == 'minimal':
        num_episodes = 5
        num_samples_settings = [1000]
    elif batch_size == 'small':
        num_episodes = 20
        num_samples_settings = [60, 1000]
    elif batch_size == 'full':
        num_episodes = 100
        num_samples_settings = [20, 60, 250, 1000]
    else:
        raise ValueError

    initialize_experiment_batch(
        batch_path = experiment_batch_dir,
        default_config     = OmegaConf.load(default_config_fname),
        controller_configs = OmegaConf.load(controller_configs_fname),
        episode_configs    = OmegaConf.load(episode_configs_fname)[0:num_episodes],
        num_samples_settings = num_samples_settings,
        save_samples = True
    )

    run_experiment_batch(experiment_batch_dir)

    analyze_experiment_batch(experiment_batch_dir)
