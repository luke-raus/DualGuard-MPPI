from dataclasses import dataclass


@dataclass
class ExperimentConfigSchema:
    # control profile params for comparison & ablation
    control_profile:                           str
    apply_safety_filter_to_final_control:      bool
    apply_safety_filter_to_nominal_trajectory: bool
    apply_safety_filter_to_samples:            bool
    cost_from_obstacles_or_BRT:                str   # TODO: should be bool or enum

    # mppi params
    mppi_samples:                     int    # number of samples to forward-simulate
    mppi_horizon:                     int    # how many dynamics steps to propagate per sample
    mppi_temperature:                 float  # higher means update incorporates votes from more samples 
    mppi_angvel_control_noise_mean:   float  # mean of control noise (rad/s)
    mppi_angvel_control_noise_stddev: float  # stddev of control noise (rad/s)
    mppi_initial_control_is_mean:     bool   # is the initial control a constant sequence of the mean control? else random

    # output params
    save_samples: bool

    # trial params
    episode_id:           int
    init_state:           list[float] # [x y theta] (m m rad)
    goal_state:           list[float] # [x y theta] (m m rad)
    goal_state_threshold: float       # (m) Refers to distance in XY plane (ignoring theta)

    # timing params
    timestep:             float       # dynamics update time interval (s)
    trial_max_duration:   float       # after this much time trial terminates (s)

    # brt & safety filter params
    brt_filename:         str
    safety_filter_type:   str
    safety_filter_value_threshold: float

    # dynamics params
    dynamics_name:        str
    state_dim:            int
    state_labels:         list[str]
    control_dim:          int
    control_labels:       list[str]
    dynamics_linvel:      float       # fixed linear velocity (m/s)
    dynamics_angvel_min:  float       # minimum angular velocity (rad/s)
    dynamics_angvel_max:  float       # maximum angular velocity (rad/s)

    # cost params
    Phi_terminal_state_cost_weights:  list[float]  # Currently unused
    Q_running_state_cost_weights:     list[float]
    action_cost_weights:              float
