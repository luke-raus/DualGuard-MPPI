from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class ExperimentResultTimestep:
    # Current time
    index   : int
    time    : float

    # Current state info
    current_state_measurement                : np.ndarray[float]  # [nx]
    current_state_brt_value                  : np.ndarray[float]
    current_state_measurement_is_unsafe      : bool
    running_cost_of_traj_incl_current_state  : float

    # MPPI starts with a nominal control sequence
    nominal_traj_controls_before   : np.ndarray[float]   # [T * nu]
    nominal_traj_states_before     : np.ndarray[float]   # [T * nx]

    # MPPI internal details
    #    Remember: Controls -> states (via simulation) -> costs, brt_values (via lookup/computation)
    sample_controls                : np.ndarray[float]   # [K * T * nu]
    sample_states                  : np.ndarray[float]   # [K * T * nx]
    sample_safety_filter_activated : np.ndarray[bool]    # [K * T]
    sample_brt_values              : np.ndarray[float]   # [K * T]
    sample_brt_theta_deriv         : np.ndarray[float]   # [K * T]
    sample_costs                   : np.ndarray[float]   # [K]
    sample_weights                 : np.ndarray[float]   # [K]

    # MPPI (or filter) results in single selected control
    control_chosen                 : np.ndarray[float]   # [nu]
    control_is_from_safety_filter  : bool

    # Controller runtime
    control_compute_time           : float

    # MPPI results in a nominal control sequence
    nominal_traj_controls_after    : np.ndarray[float]   # [T * nu]
    nominal_traj_states_after      : np.ndarray[float]   # [T * nx]


@dataclass(slots=True)
class ExperimentResultSummary:
    crashed:        bool
    goal_reached:   bool
    time_elapsed:   float
    total_cost:     float
    terminal_state: list[float]
    control_compute_time_avg: float
    control_compute_time_std: float


class ExperimentResult:

    def __init__(self):
        self.timesteps: list[ExperimentResultTimestep] = []
        self.summary:   ExperimentResultSummary        = None

    def capture_timestep(self, timestep:ExperimentResultTimestep) -> None:
        # Always includes sample data; could lead to memory issues for very large experiments.
        self.timesteps.append(timestep)

    def capture_summary(self, summary:ExperimentResultSummary) -> None:
        self.summary = summary

    def get_total_trajectory(self) -> list:
        trajectory = self.get_attribute_across_timesteps('current_state_measurement')
        # Since above states are measured at the start of each controller iteration, add final state
        trajectory.append(self.summary.terminal_state)
        return trajectory

    def get_attribute_across_timesteps(self, attribute:str) -> list:
        return [getattr(step, attribute) for step in self.timesteps]
