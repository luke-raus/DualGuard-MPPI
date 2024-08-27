from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
import h5py


class ExperimentResult:
    def __init__(self, experiment_dir:str, save_samples:bool=True):
        # self.experiment_dir = experiment_dir
        self.summary_fname = Path(experiment_dir) / 'result_summary.yaml'
        self.details_fname = Path(experiment_dir) / 'result_details.hdf5'

        self.save_samples = save_samples

        self.timesteps = []
        self.summary = None

    def capture_timestep(
        self,
        # What we know about current state
        index: int,
        time: float,
        current_state_measurement: np.ndarray,
        current_state_measurement_is_unsafe: bool,
        running_cost_of_traj_incl_current_state: float,
        # MPPI starts with a nominal control sequence
        nominal_traj_controls_before: np.ndarray,
        nominal_traj_states_before: np.ndarray,
        # MPPI internal details: optional to save to file,
        #    but must include them in case we want to save at some point
        #    Remember: Controls -> states (via simulation) -> costs, brt_values (via lookup/computation)
        sample_controls: np.ndarray,
        sample_states: np.ndarray,
        sample_safety_filter_activated: np.ndarray,
        sample_costs: np.ndarray,
        sample_brt_values: np.ndarray,
        sample_brt_theta_deriv: np.ndarray,
        # MPPI results in a nominal control sequence
        nominal_traj_controls_after: np.ndarray,
        nominal_traj_states_after: np.ndarray,
        # Controller runtime
        control_compute_time: float,
        # MPPI (or filter) results in single selected control
        control_chosen: np.ndarray,
        control_overridden_by_safety_filter: bool,
    ) -> None:
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

        # Save timesteps as groups in hdf5 file
        with h5py.File(self.details_fname, 'w') as hdf_file:
            for timestep in self.timesteps:
                index = timestep['index']
                group = hdf_file.create_group(f'step_{index}')
                for key, data in timestep.items():
                    if (not self.save_samples) and (key not in ['sample_controls', 'sample_states', 'sample_safety_filter_activated',
                                   'sample_costs', 'sample_brt_values', 'sample_brt_theta_deriv']):
                        group.create_dataset(key, data=data)

            # Consider saving overall state trajectory sequence in its own dataset

    def load_summary(self) -> None:
        self.result_summary = OmegaConf.load(self.summary_fname)

    def load_timestep(self, timestep_index:int) -> dict:
        step_data = {}
        with h5py.File(self.details_fname) as f:
            group = f[f'step_{timestep_index}']
            for key in group.keys():
                step_data[key] = group[key][()]
        return step_data

    def get_num_timesteps(self) -> int:
        with h5py.File(self.details_fname, 'r') as f:
            # Assuming each top-level entry is a timestep
            timesteps = f.keys()  # if .... == 'step_'
            return len(timesteps)
