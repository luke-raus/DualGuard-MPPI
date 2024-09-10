from omegaconf import OmegaConf
from pathlib import Path
import numpy as np   # only needed for type-checking
import h5py


class ExperimentResult:
    def __init__(self, experiment_dir, save_samples:bool=True):
        self.experiment_dir = experiment_dir

        self.config_fname = Path(experiment_dir) / 'config.yaml'

        self.summary_fname = Path(experiment_dir) / 'result_summary.yaml'
        self.details_fname = Path(experiment_dir) / 'result_details.hdf5'

        self.save_samples = save_samples

        self.timesteps = []
        self.summary = None
        self.config = None

    def capture_timestep(
        self,
        index: int,
        time: float,
        current_state_measurement: np.ndarray,
        current_state_measurement_is_unsafe: bool,
        running_cost_of_traj_incl_current_state: float,
        nominal_traj_controls_before: np.ndarray,
        nominal_traj_states_before: np.ndarray,
        sample_controls: np.ndarray,
        sample_states: np.ndarray,
        sample_safety_filter_activated: np.ndarray,
        sample_costs: np.ndarray,
        sample_weights: np.ndarray,
        sample_brt_values: np.ndarray,
        sample_brt_theta_deriv: np.ndarray,
        nominal_traj_controls_after: np.ndarray,
        nominal_traj_states_after: np.ndarray,
        control_compute_time: float,
        control_chosen: np.ndarray,
        control_overridden_by_safety_filter: bool,
    ) -> None:
        # See results_config.yaml for explanation/grouping of parameters

        # Capture all arguments (since they're mandatory!)
        #    in a dict with locals(), but exclude 'self'
        timestep = locals()
        timestep.pop("self")
        self.timesteps.append(timestep)

    def capture_summary(
        self,
        crashed: bool,
        goal_reached: bool,
        time_elapsed: float,
        total_cost: float,
        terminal_state: np.ndarray,
        control_compute_time_avg: float,
        control_compute_time_std: float,
    ) -> None:
        summary = locals()
        summary.pop("self")  # Exclude 'self'
        # For OmegaConf compatibility
        summary["terminal_state"] = [float(x) for x in summary["terminal_state"]]
        self.summary = OmegaConf.create(summary)

    def save_to_files(self) -> None:

        if self.summary is None:
            raise ValueError("Must use capture final summary before saving!")
        if len(self.timesteps) == 0:
            raise ValueError("Must capture at least one timestep before saving!")

        # Save summary as human-readable .yaml
        OmegaConf.save(self.summary, f=self.summary_fname)

        sample_dataset_names = ['sample_controls', 'sample_states', 'sample_safety_filter_activated',
                                'sample_costs', 'sample_weights', 'sample_brt_values', 'sample_brt_theta_deriv']
        # Save timesteps as groups in hdf5 file
        with h5py.File(self.details_fname, 'w') as hdf_file:
            for timestep in self.timesteps:
                index = timestep['index']
                group = hdf_file.create_group(f'step_{index}')
                for key, data in timestep.items():
                    if not (self.save_samples==False and (str(key) in sample_dataset_names)):
                        group.create_dataset(key, data=data)

            # Save overall state trajectory sequence in its own dataset
            state_trajectory = [ step['current_state_measurement'] for step in self.timesteps ]
            # Since the above states are measured at the start of a controller iteration, append last state
            state_trajectory.append(self.summary.terminal_state)
            hdf_file.create_dataset('state_trajectory', data=state_trajectory)

    def get_summary(self) -> dict:
        self.summary = OmegaConf.load(self.summary_fname)
        return self.summary

    def get_config(self) -> dict:
        self.config = OmegaConf.load(self.config_fname)
        return self.config

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
    