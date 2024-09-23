from omegaconf import OmegaConf
from pathlib import Path
import h5py
import numpy as np

from experiment_result import ExperimentResult


class ExperimentStorage:

    def __init__(self, experiment_dir):

        self.experiment_dir = experiment_dir

        self.config_fname  = Path(experiment_dir)/'config.yaml'
        self.summary_fname = Path(experiment_dir)/'result_summary.yaml'
        self.details_fname = Path(experiment_dir)/'result_details.hdf5'

        self.summary = None
        self.config = None

        self.is_complete = False

    def save_config(self, config:OmegaConf) -> None:

        self.config = config
        OmegaConf.save(self.config, f=self.config_fname)

    def save_results(self, result:ExperimentResult) -> None:

        if len(result.timesteps) == 0:
            raise ValueError("Must capture at least one timestep before saving!")

        # Save summary in human-readable YAML file
        self.summary = result.summary
        OmegaConf.save(self.summary, f=self.summary_fname)

        # Save timesteps as groups in hdf5 file
        with h5py.File(self.details_fname, 'w') as hdf_file:
            for timestep in result.timesteps:
                index = timestep['index']
                group = hdf_file.create_group(f'step_{index}')
                for key, data in timestep.items():
                    group.create_dataset(key, data=data)

            # Save overall state trajectory sequence in its own dataset
            state_trajectory = [ step['current_state_measurement'] for step in result.timesteps ]
            # Since the above states are measured at the start of a controller iteration, append last state
            state_trajectory.append(self.summary.terminal_state)
            hdf_file.create_dataset('state_trajectory', data=state_trajectory)

    def get_config(self) -> dict|None:
        if self.config is not None:
            return self.config
        try:
            self.config = OmegaConf.load(self.config_fname)
            return self.config
        except FileNotFoundError:
            return None

    def get_summary(self) -> dict|None:
        if self.summary is not None:
            return self.summary
        try:
            self.summary = OmegaConf.load(self.summary_fname)
            return self.summary
        except FileNotFoundError:
            return None

    def get_all_experiment_info(self) -> dict:
        self.get_summary()
        self.get_config()
        return {'path':str(self.experiment_dir), **self.config, **self.summary}

    def get_overall_trajectory(self) -> np.ndarray:
        with h5py.File(self.details_fname, 'r') as f:
            traj = f['state_trajectory'][:]
        return traj

    def get_timestep_data(self, step_index:int) -> dict:
        step_data = {}
        with h5py.File(self.details_fname, 'r') as f:
            group = f[f'step_{step_index}']
            for key in group.keys():
                step_data[key] = group[key][()]
        return step_data

    def get_num_timesteps(self) -> int:
        # Alternatively, could return len of overall_trajectory - 1
        with h5py.File(self.details_fname, 'r') as f:
            timesteps = [k for k in f.keys() if k.startswith('step_')]
        return len(timesteps)

    def get_environment_path(self) -> Path:
        self.get_config()
        return self.config.brt_filename
