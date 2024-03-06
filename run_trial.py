import time
import numpy as np


def run_trial(system, map, controller, max_timesteps, safety_filter, save_samples=False, diagnostics=False):
    """
    safety_filter is boolean of whether to invoke reachability-based safety filter
    """

    goal_state = map.goal_state
    goal_state_threshold = 0.1

    expr_data = {
        't':               [0.],
        'actual_state':    [system.state],
        'actual_action':   [],
        'cost_of_step':    [],
        'running_cost':    [],
        'sample_costs':    [],
        'sample_weights':  [],
        'nom_states':      [],
        'nom_actions_before': [],
        'nom_actions_after':  [],
        'safety_being_violated': [],
        'safety_filter_activated': [],
        'computation_time': [],
        'goal_reached': False,
        'crashed': False,
        'total_traj_cost': None
    }
    if save_samples:
        expr_data['sampled_states'] = []
        expr_data['sampled_actions'] = []
    if diagnostics:
        expr_data['sample_brt_values'] = []
        expr_data['sample_brt_theta_deriv'] = []
        expr_data['sample_safety_filter'] = []

    running_cost = 0
    for i in range(max_timesteps):

        expr_data['nom_actions_before'].append(controller.U)

        # Check whether current system state is unsafe
        safety_being_violated = bool(map.check_brt_collision(system.state))

        # Run MPPI controller anyways, but don't necessarily pass action to state

        timer_start = time.perf_counter()
        mppi_action = controller.command(system.state)

        potential_next_state = system.next_states(
            system.state, mppi_action)
        next_state_unsafe = bool(map.check_brt_collision( np.expand_dims(potential_next_state, axis=0) ))
        # If relevant, override MPPI-chosen control action with safety control
        safety_filter_activated = ( (safety_being_violated or next_state_unsafe) and safety_filter)
        if safety_filter_activated:
            # Safety filter activated! Choose action using BRT safety controller
            action = map.get_brt_safety_control( np.expand_dims(system.state, axis=0) ).squeeze(axis=0)
        else:
            action = mppi_action

        cost, temp = map.get_state_progress_and_obstacle_costs( np.expand_dims(system.state, axis=0), np.expand_dims(mppi_action, axis=0))
        cost = cost.squeeze(axis=0)
        running_cost += cost

        timer_elapsed = time.perf_counter() - timer_start

        # Pass action to system
        system.update_true_state(action)

        print_progress = True
        if print_progress:
            #if i % 10 == 0:
            print(f"controller iteration {i}, time elapsed: {timer_elapsed}")

        state = system.state

        expr_data['t'].append(i * system.timestep)
        expr_data['actual_state'].append(state)
        expr_data['actual_action'].append(action)
        expr_data['cost_of_step'].append(cost)
        expr_data['running_cost'].append(running_cost)
        expr_data['sample_costs'].append(controller.cost_total)
        expr_data['sample_weights'].append(controller.omega)
        expr_data['nom_states'].append(controller.nominal_trajectory)
        expr_data['nom_actions_after'].append(controller.U)
        expr_data['safety_being_violated'].append(safety_being_violated)
        expr_data['safety_filter_activated'].append(safety_filter_activated)
        expr_data['computation_time'].append(timer_elapsed)
        if save_samples:
            expr_data['sampled_states'].append(controller.sampled_states)
            expr_data['sampled_actions'].append(
                controller.sampled_actions)
        if diagnostics:
            expr_data['sample_brt_values'].append(controller.sample_brt_values)
            expr_data['sample_brt_theta_deriv'].append(
                controller.sample_brt_theta_deriv)
            expr_data['sample_safety_filter'].append(
                controller.sample_safety_filter)

        # Check if we've reached goal state within threshold; if so, end trial
        dist_to_goal_state = float(
            (state[0]-goal_state[0])**2 + (state[1]-goal_state[1])**2)
        if (dist_to_goal_state < goal_state_threshold**2):
            expr_data['goal_reached'] = True
            print('goal reached')
            return expr_data

        # Check if we've collided with an obstacle; if so, end trial
        if map.check_obs_collision(np.expand_dims(state, axis=0)):
            expr_data['crashed'] = True
            print('crashed')
            return expr_data

    # We've run out of experiment timesteps without finishing or crashing, so end trial
    return expr_data
