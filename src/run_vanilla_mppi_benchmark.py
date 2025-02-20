from omegaconf import OmegaConf
from pathlib import Path
import json

from experiment_storage import ExperimentStorage
from experiment_runner import ExperimentRunner


experiments_path = Path('experiments_sep_25_benchmark_vanilla')

default_config_fname    = Path('config') / 'default_config.yaml'
control_profiles_fname  = Path('config') / 'control_profiles.yaml'
trial_state_pairs_fname = Path('config') / 'trial_state_pairs.json'

num_samples_opts = [500]

angvel_stddev_opts = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

mppi_temp_opts =     [0.2, 0.6, 1.0, 2.6, 5.0, 8.0]

num_trials_per_config = 50

save_samples = True


with open(trial_state_pairs_fname, 'r') as f:
    init_goal_state_pairs = json.load(f)
    init_goal_state_pairs = init_goal_state_pairs[:num_trials_per_config]

control_profiles = OmegaConf.load(control_profiles_fname)


# === Set up experiment configurations ===

for num_samples in num_samples_opts:
    for trial_ind, init_goal_state_pair in enumerate(init_goal_state_pairs):
        for angvel_stddev in angvel_stddev_opts:
            for mppi_temp in mppi_temp_opts:
                for control_ind, (profile_name, profile_settings) in enumerate(control_profiles.items()):

                    if not control_ind == 0:
                        continue

                    # Load default config
                    config = OmegaConf.load(default_config_fname)

                    # Override defaults with this experiment's settings
                    num_settings = len(config)
                    config.update(profile_settings)
                    config.control_profile = profile_name

                    config.mppi_temperature = mppi_temp
                    config.mppi_angvel_control_noise_stddev = angvel_stddev

                    config.mppi_samples = num_samples
                    config.init_state   = init_goal_state_pair['init']
                    config.goal_state   = init_goal_state_pair['goal']
                    config.save_samples = save_samples
                    assert(len(config) == num_settings), "Extraneous config options have been added!"

                    ang_std_str = str(angvel_stddev).replace('.', ',')
                    temp_str    = str(mppi_temp).replace('.', ',')
                    # Set & create experiment directory, then save config file
                    exp_fname = f"exp_vanilla500samp_ustd-{ang_std_str}_temp-{temp_str}_trial-{trial_ind:03}"

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
