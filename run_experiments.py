from controller_mppi import MPPI
from dubins_environment import ClutteredMap
from dubins_dynamics import DubinsCarFixedVel
from run_trial import run_trial
from experiment_result import ExperimentResult

# Used to load a dot-accessible config dict from .yaml file (& cmd-line args)
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np


def run_experiment(config_filename, results_filepath):

    config = OmegaConf.load(config_filename)

    system = DubinsCarFixedVel(
        config.timestep,
        config.dynamics_linvel,
        np.array(config.init_state)
    )

    environment = ClutteredMap(
        config.Q_running_state_cost_weights,
        config.Phi_terminal_state_cost_weights,
        config.action_cost_weights,
        init_state = config.init_state,
        goal_state = config.goal_state,
        brt_file   = config.brt_filename,
        brt_value_threshold = config.safety_filter_value_threshold,
        cost_type = config.cost_from_obstacles_or_BRT,
    )

    controller = MPPI(
        system.next_states,
        environment.get_state_progress_and_obstacle_costs,
        nx          = system.nx,
        num_samples = config.mppi_samples,
        horizon     = config.mppi_horizon,
        noise_mu    = config.mppi_angvel_control_noise_mean,
        noise_sigma = (config.mppi_angvel_control_noise_stddev)**2,
        u_min       = config.dynamics_angvel_min,
        u_max       = config.dynamics_angvel_max,
        # U_init = torch.zeros(MPPI_HORIZON, n_inputs),
        terminal_state_cost = environment.get_terminal_state_cost,
        noise_abs_cost      = True,
        lambda_             = config.mppi_temperature,
        filter_nom_traj     = config.apply_safety_filter_to_nominal_trajectory,
        filter_samples      = config.apply_safety_filter_to_samples,
        brt_safety_query    = environment.check_brt_collision,
        brt_opt_ctrl_query  = environment.get_brt_safety_control,
        brt_value_query     = environment.get_brt_value,
        brt_theta_deriv_query = environment.get_brt_theta_deriv,
        diagnostics = False,
    )

    result = run_trial(
        system, environment, controller,
        max_timesteps = int(config.trial_max_duration / config.timestep),
        safety_filter = config.apply_safety_filter_to_samples,
        results_filepath = results_filepath,
        save_samples = config.save_samples,
    )
    result.save_to_files()

    print(f'Experiment complete')


def save_experiment_results(exp_name, overview, trajectory, sample_details):
    base_dir = exp_name #Path('experiments') / exp_name
    base_dir.mkdir(parents=True, exist_ok=True)
    overview_fname       = base_dir / 'result_overview.yaml'
    trajectory_fname     = base_dir / 'result_trajectory.csv'
    sample_details_fname = base_dir / 'result_sample_details.XXX'

    # Save overview as .yaml
    OmegaConf.save(OmegaConf.create(overview), f=overview_fname)

    # Save trajectory as .csv
    trajectory.to_csv(trajectory_fname, index=False)

    # Save sample_details
    pass
