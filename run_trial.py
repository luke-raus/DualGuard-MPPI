import time
import numpy as np
import pandas as pd

def run_trial(system, map, controller, max_timesteps, safety_filter, save_samples=False, diagnostics=False):
    """
    safety_filter is boolean of whether to invoke reachability-based safety filter
    """

    goal_state = map.goal_state
    goal_state_threshold = 0.1

    goal_reached = False
    crashed = False

    running_cost = 0

    history = []
    for i in range(max_timesteps):

        # expr_data['nom_actions_before'].append(controller.U)

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

        state = system.state

        history.append( {
            't': i * system.timestep,
            'state': state,
            'control_taken': action,
            'cost_of_step': cost,
            'running_cost': running_cost,
            'safety_being_violated': safety_being_violated,
            'safety_filter_activated': safety_filter_activated,
            'control_computation_time': timer_elapsed,
        } )

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

    # We've run out of experiment timesteps without finishing or crashing, so end trial
    history = pd.DataFrame(history)
    overview = {
        'crashed':       crashed,
        'goal_reached':  goal_reached,
        'time_elapsed':  i * system.timestep,
        'total_cost':    round(running_cost, 5),
        # these calculations give np.float64, which omegaconf dislikes
        'control_compute_time_avg': round(float(history['control_computation_time'].mean()), 6),
        'control_compute_time_std': round(float(history['control_computation_time'].std()),  6),
    }
    sample_details = None

    return overview, history, sample_details
