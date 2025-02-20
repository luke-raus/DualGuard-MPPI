from omegaconf import OmegaConf
import numpy as np   # only needed for type-checking


class ExperimentResult:

    def __init__(self, save_samples:bool=True):

        self.timesteps = []
        self.summary = None

        self.save_samples = save_samples

        self.sample_related_keys = [
            'sample_controls',
            'sample_states',
            'sample_safety_filter_activated',
            'sample_costs',
            'sample_weights',
            'sample_brt_values',
            'sample_brt_theta_deriv'
        ]

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
        #    in a dict with locals(), but exclude 'self' and possible sample-related
        timestep = locals()
        del timestep["self"]

        if not self.save_samples:
            for sample_key in self.sample_related_keys:
                del timestep[sample_key]

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
        del summary["self"]
        # For OmegaConf compatibility
        summary["terminal_state"] = [float(x) for x in summary["terminal_state"]]
        self.summary = OmegaConf.create(summary)

    def get_total_trajectory(self) -> list:
        trajectory = self.get_attribute_across_timesteps('current_state_measurement')
        # Since above states are measured at the start of each controller iteration, add final state
        trajectory.append(self.summary.terminal_state)
        return trajectory

    def get_attribute_across_timesteps(self, attribute:str) -> list:
        return [step[attribute] for step in self.timesteps]