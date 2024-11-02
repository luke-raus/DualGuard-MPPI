from omegaconf import OmegaConf
from pathlib import Path

from experiment_storage import ExperimentStorage
from experiment_runner import ExperimentRunner
from experiment_config import ExperimentConfigSchema

#FIXME:update to final name
experiments_path = Path('exp_shield')

default_config_fname     = Path('config') / 'default_config.yaml'
controller_configs_fname = Path('config') / 'control_profiles.yaml'
episode_configs_fname    = Path('config') / 'episode_params.yaml'

#FIXME: Keep the deafults as low as possible, new users want to check if it works before commiting for a full day stuck in sim
#mppi_samples_settings = [20, 36, 60, 100, 250, 500, 1000]
mppi_samples_settings = [20]
#num_episodes = 100
num_episodes = 3

save_samples = True


config_schema = OmegaConf.structured(ExperimentConfigSchema)
default_config = OmegaConf.load(default_config_fname)
# Will error if default config does not conform to schema
structured_default_config = OmegaConf.merge(config_schema, default_config)

episode_configs    = OmegaConf.load(episode_configs_fname)[0:num_episodes]
controller_configs = OmegaConf.load(controller_configs_fname)

# === Set up experiment configurations ===

for mppi_samples in mppi_samples_settings:
    for episode_config in episode_configs:
        for control_ind, controller_config in enumerate(controller_configs):

            # Update default config with this experiment's settings.
            # Since default is structured, this errors if we try to add new keys.
            config = OmegaConf.merge(structured_default_config, episode_config, controller_config)
            config.mppi_samples = mppi_samples
            config.save_samples = save_samples

            # Set & create experiment directory, then save config file
            exp_fname = f"exp_samples-{mppi_samples:04}_ep-{episode_config['episode_id']:03}_control-{control_ind}"

            experiment_dir = experiments_path / exp_fname
            experiment_dir.mkdir(parents=True, exist_ok=False)
            # May want to let user know that we're avoiding overriding existing experiment dirs
            OmegaConf.save(config, f=experiment_dir/'config.yaml')


# === Run experiments ===

for experiment_dir in sorted(experiments_path.iterdir()):

    print(f'Running experiment {experiment_dir}')

    experiment_storage = ExperimentStorage(experiment_dir)
    experiment = ExperimentRunner(experiment_storage)
    experiment.initialize()
    experiment.run_and_save()

print('All done!')
