from controller_mppi import MPPI
from dubins_environment import ClutteredMap
from dubins_dynamics import DubinsCarFixedVel

from experiment_result import ExperimentResult
from experiment_storage import ExperimentStorage

import time
import numpy as np


class ExperimentRunner:

    def __init__(self, storage:ExperimentStorage):
        self.storage = storage
        self.config = storage.get_config()

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
            brt_fname  = self.config.brt_filename,
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
            random_seed = int(self.config.init_state[0]**2 * 1e6),
        )

    def run(self) -> ExperimentResult:

        config = self.config
        system = self.system
        map = self.environment
        controller = self.controller

        result = ExperimentResult(save_samples=config.save_samples)


        max_timesteps = int(config.trial_max_duration / config.timestep)
        safety_filter_enabled = config.apply_safety_filter_to_final_chosen_control,

        goal_reached, crashed = False, False
        running_cost = 0

        for i in range(max_timesteps):

            # Capture nominal trajectory before MPPI runs
            nominal_traj_controls_before = controller.U
            nominal_traj_states_before = controller.nominal_trajectory

            # Capture state before controller runs
            measured_state = system.state
            measured_state_is_unsafe = bool(map.check_brt_collision(system.state))

            # Run MPPI controller anyways, but don't necessarily pass action to state
            timer_start = time.perf_counter()

            """
            selected_control, step_cost = contoller.get_control(state)
            """

            mppi_action = controller.command(system.state)

            potential_next_state = system.next_states(system.state, mppi_action)
            next_state_unsafe = bool(map.check_brt_collision( np.expand_dims(potential_next_state, axis=0) ))

            # If relevant, override MPPI-chosen control action with safety control
            safety_filter_activated = ( (measured_state_is_unsafe or next_state_unsafe) and safety_filter_enabled)
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

            print_progress = False
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
                sample_weights = controller.omega,
                sample_brt_values = controller.sample_brt_values,
                sample_brt_theta_deriv = controller.sample_brt_theta_deriv,
                nominal_traj_controls_after = controller.U,
                nominal_traj_states_after = controller.nominal_trajectory,
                control_compute_time = timer_elapsed,
                control_chosen = action,
                control_overridden_by_safety_filter = safety_filter_activated
            )

            # If we're close enough to goal, end trial successfully
            if self.environment.get_dist_to_goal(system.state) < config.goal_state_threshold:
                goal_reached = True
                print('goal reached')
                break

            # Check if we've collided with an obstacle; if so, end trial
            if map.check_obs_collision(np.expand_dims(system.state, axis=0)):
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

    def run_and_save(self) -> None:
        result = self.run()
        self.storage.save_results(result)