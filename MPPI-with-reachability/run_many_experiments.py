from omegaconf import OmegaConf
from pathlib import Path
import json

from experiment_storage import ExperimentStorage
from experiment_runner import ExperimentRunner


experiments_path = Path('experiments_text')

default_config_fname    = Path('config') / 'default_config.yaml'
control_profiles_fname  = Path('config') / 'control_profiles.yaml'
trial_state_pairs_fname = Path('config') / 'trial_state_pairs.json'

num_samples_opts = [20, 36, 60, 100, 250, 500, 1000]

num_trials_per_config = 100

save_samples = True


with open(trial_state_pairs_fname, 'r') as f:
    init_goal_state_pairs = json.load(f)
    init_goal_state_pairs = init_goal_state_pairs[:num_trials_per_config]

control_profiles = OmegaConf.load(control_profiles_fname)


# === Set up experiment configurations ===

for num_samples in num_samples_opts:
    for trial_ind, init_goal_state_pair in enumerate(init_goal_state_pairs):
        for control_ind, (profile_name, settings) in enumerate(control_profiles.items()):

            # Load default config
            config = OmegaConf.load(default_config_fname)

            # Override defaults with this experiment's settings
            config.update(settings)
            config.control_profile = profile_name
            config.mppi_samples = num_samples
            config.init_state   = init_goal_state_pair['init']
            config.goal_state   = init_goal_state_pair['goal']
            config.save_samples = save_samples

            # Set & create experiment directory, then save config file
            exp_fname = f"exp_samples-{num_samples:04}_trial-{trial_ind:03}_control-{control_ind}"

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
