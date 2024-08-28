from controller_mppi import MPPI
from dubins_environment import ClutteredMap
from dubins_dynamics import DubinsCarFixedVel
from experiment_result import ExperimentResult

import time
import numpy as np
from pathlib import Path

# Used to load a dot-accessible config dict from .yaml file (& cmd-line args)
from omegaconf import OmegaConf


class Experiment:

    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.config_fname  = self.experiment_dir / 'config.yaml'
        self.summary_fname = self.experiment_dir / 'result_summary.yaml'
        self.details_fname = self.experiment_dir / 'result_details.hdf5'

        # Load config
        self.config = OmegaConf.load(self.config_fname)

    def initialize(self) -> None:

        self.system = DubinsCarFixedVel(
            self.config.timestep,
            self.config.dynamics_linvel,
            np.array(self.config.init_state)
        )

        self.environment = ClutteredMap(
            self.config.Q_running_state_cost_weights,
            self.config.Phi_terminal_state_cost_weights,
            self.config.action_cost_weights,
            init_state = self.config.init_state,
            goal_state = self.config.goal_state,
            brt_file   = self.config.brt_filename,
            brt_value_threshold = self.config.safety_filter_value_threshold,
            cost_type = self.config.cost_from_obstacles_or_BRT,
        )

        self.controller = MPPI(
            self.system.next_states,
            self.environment.get_state_progress_and_obstacle_costs,
            nx          = self.system.nx,
            num_samples = self.config.mppi_samples,
            horizon     = self.config.mppi_horizon,
            noise_mu    = self.config.mppi_angvel_control_noise_mean,
            noise_sigma = (self.config.mppi_angvel_control_noise_stddev)**2,
            u_min       = self.config.dynamics_angvel_min,
            u_max       = self.config.dynamics_angvel_max,
            U_init_is_mean      = self.config.mppi_initial_control_is_mean,            
            terminal_state_cost = self.environment.get_terminal_state_cost,
            noise_abs_cost      = True,
            lambda_             = self.config.mppi_temperature,
            filter_nom_traj     = self.config.apply_safety_filter_to_nominal_trajectory,
            filter_samples      = self.config.apply_safety_filter_to_samples,
            brt_safety_query    = self.environment.check_brt_collision,
            brt_opt_ctrl_query  = self.environment.get_brt_safety_control,
            brt_value_query     = self.environment.get_brt_value,
            brt_theta_deriv_query = self.environment.get_brt_theta_deriv,
            diagnostics = False,
        )

    def run_and_save(self) -> None:

        result = self.run_trial()
        result.save_to_files()

        print(f'Experiment complete')


    def run_trial(self) -> ExperimentResult:
        """
        safety_filter is boolean of whether to invoke reachability-based safety filter
        """

        config = self.config
        system = self.system
        map = self.environment
        controller = self.controller

        max_timesteps = int(config.trial_max_duration / config.timestep)
        safety_filter = config.apply_safety_filter_to_samples,

        goal_state = map.goal_state
        goal_state_threshold = 0.1

        goal_reached = False
        crashed = False

        running_cost = 0

        result = ExperimentResult(self.experiment_dir, save_samples=config.save_samples)

        for i in range(max_timesteps):

            # Capture nominal trajectory before MPPI runs
            nominal_traj_controls_before = controller.U
            nominal_traj_states_before = controller.nominal_trajectory

            # Capture state before controller runs
            measured_state = system.state
            measured_state_is_unsafe = bool(map.check_brt_collision(system.state))

            # Run MPPI controller anyways, but don't necessarily pass action to state
            timer_start = time.perf_counter()
            mppi_action = controller.command(system.state)

            potential_next_state = system.next_states(
                system.state, mppi_action)
            next_state_unsafe = bool(map.check_brt_collision( np.expand_dims(potential_next_state, axis=0) ))
            # If relevant, override MPPI-chosen control action with safety control
            safety_filter_activated = ( (measured_state_is_unsafe or next_state_unsafe) and safety_filter)
            if safety_filter_activated:
                # Safety filter activated! Choose action using BRT safety controller
                action = map.get_brt_safety_control( np.expand_dims(system.state, axis=0) ).squeeze(axis=0)
            else:
                action = mppi_action

            cost, _ = map.get_state_progress_and_obstacle_costs( np.expand_dims(system.state, axis=0), np.expand_dims(mppi_action, axis=0))
            cost = float(cost.squeeze(axis=0))
            running_cost += cost

            timer_elapsed = time.perf_counter() - timer_start

            # Pass action to system
            system.update_true_state(action)

            print_progress = True
            if print_progress:
                #if i % 10 == 0:
                print(f"controller iteration {i}, time elapsed: {timer_elapsed:.6f}")

            result.capture_timestep(
                index = i,
                time = i * system.timestep,
                current_state_measurement = measured_state,
                current_state_measurement_is_unsafe = measured_state_is_unsafe,
                running_cost_of_traj_incl_current_state = running_cost,
                nominal_traj_controls_before = nominal_traj_controls_before,
                nominal_traj_states_before = nominal_traj_states_before,
                sample_controls = controller.sampled_actions,
                sample_states = controller.sampled_states,
                sample_safety_filter_activated = controller.sample_safety_filter,
                sample_costs = controller.cost_total,
                sample_brt_values = controller.sample_brt_values,
                sample_brt_theta_deriv = controller.sample_brt_theta_deriv,
                nominal_traj_controls_after = controller.U,
                nominal_traj_states_after = controller.nominal_trajectory,
                control_compute_time = timer_elapsed,
                control_chosen = action,
                control_overridden_by_safety_filter = safety_filter_activated
            )

            state = system.state

            # if close_to_goal():
            # Check if we've reached goal state within threshold; if so, end trial
            dist_to_goal_state_sq = float(
                (state[0]-goal_state[0])**2 + (state[1]-goal_state[1])**2)
            if (dist_to_goal_state_sq < goal_state_threshold**2):
                goal_reached = True
                print('goal reached')
                break

            # Check if we've collided with an obstacle; if so, end trial
            if map.check_obs_collision(np.expand_dims(state, axis=0)):
                crashed = True
                print('crashed')
                break

        result.capture_summary(
            crashed = crashed,
            goal_reached = goal_reached,
            time_elapsed = i*system.timestep,
            total_cost = round(running_cost, 5),
            terminal_state = system.state,
            control_compute_time_avg = False, # round(float(history['control_computation_time'].mean()), 6),
            control_compute_time_std = False, # round(float(history['control_computation_time'].std()),  6)
        )

        return result
