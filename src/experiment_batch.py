from omegaconf import OmegaConf
from pathlib import Path

from experiment_storage import ExperimentStorage
from experiment_runner import ExperimentRunner
from experiment_config import ExperimentConfigSchema


def initialize_experiment_batch(
        batch_path:Path,
        default_config:OmegaConf,
        controller_configs:list[OmegaConf],
        episode_configs:list[OmegaConf],
        num_samples_settings:list[int],
        save_samples:bool) -> None:
    """
    Creates a subdirectory with config file for each experiment to run in batch.
    """

    # Ensure passed default config conforms to schema (otherwise this will error).
    config_schema = OmegaConf.structured(ExperimentConfigSchema)
    structured_default_config = OmegaConf.merge(config_schema, default_config)

    print(f'Initializing experiments in: {batch_path.absolute()}')
    num_experiments = 0

    for mppi_samples in num_samples_settings:
        for episode_config in episode_configs:
            for control_ind, controller_config in enumerate(controller_configs):

                # Update default config with this experiment's settings.
                # Since default is structured, this errors if we try to add new keys.
                config = OmegaConf.merge(structured_default_config, episode_config, controller_config)
                config.mppi_samples = mppi_samples
                config.save_samples = save_samples

                controller_name = controller_config['control_profile']
                # Set & create experiment directory, then save config file
                exp_fname = f'exp_samples-{mppi_samples:04}_ep-{episode_config['episode_id']:03}_control-{control_ind}-{controller_name}'

                experiment_dir = batch_path / exp_fname
                experiment_dir.mkdir(parents=True, exist_ok=False)
                # May want to let user know that we're avoiding overriding existing experiment dirs

                storage = ExperimentStorage(experiment_dir)
                storage.save_config(config)

                num_experiments += 1

    print(f'==== Initialized {num_experiments} experiments. ====')


def run_experiment_batch(batch_path:Path) -> None:
    print('Running experiments...')
    for experiment_dir in sorted(batch_path.iterdir()):

        print(f'Running experiment: {experiment_dir}')
        experiment_storage = ExperimentStorage(experiment_dir)
        experiment = ExperimentRunner(experiment_storage)
        experiment.initialize()
        experiment.run_and_save()

    print('==== Finished experiments. ====')
    print(f'Results are saved to: {batch_path.absolute()} ====')
