from controller_mppi import MPPI
from dubins_environment import ClutteredMap
from dubins_dynamics import DubinsCarFixedVel
from run_trial import run_trial
import cProfile

import numpy as np
import pickle


if __name__ == "__main__":

    #profiler = cProfile.Profile()

    experiment_time_horizon = 15   # sec

    timestep = 0.02
    max_timesteps = 30 #round(experiment_time_horizon / timestep)

    N_MPPI_SAMPLES = 1000

    # Dyn. sys. definition
    angvel_min, angvel_max = (-6.0, 6.0)   # rad/s
    angvel_noise_mean = 0.0
    linvel = 4.0                           # m/s

    angvel_noise_stddev = 8.0

    # unused currently...
    Phi_terminal_state_cost_weights = np.array([0., 0., 0.])
    Q_running_state_cost_weights    = np.array([1., 1., 0.])
    action_cost_weights = 0.2

    state_pairs = pickle.load(open("config_data/state_pairs_outside_disturbed_brt_dec_29.pkl", "rb"))

    map_data = pickle.load(open("config_data/new_map_aug_12.pkl", "rb"))
    walls = 5.0

    brt_file = "config_data/brt_dubin_new_map_disturbed_aug_16_fixed_init_value.mat"
    brt_value_threshold = 0.01

    placeholder_init_state = np.array([0., 0., 0.])

    system = DubinsCarFixedVel(timestep,
                               linvel,
                               placeholder_init_state)
    n_states = system.nx
    n_inputs = system.nu

    horizon = 25
    mppi_temperature = 5.0

    trial = 1
    filter_samples, safety_filter, cost_type = False, False, 'obs'
    #filter_samples, safety_filter, cost_type = True, False, 'obs'
    #filter_samples, safety_filter, cost_type = True, False, 'brt'
    #filter_samples, safety_filter, cost_type = False, False, 'brt'

    filter_nom_traj = False

    init_state = np.array(state_pairs['init'][trial])
    goal_state = np.array(state_pairs['goal'][trial])

    system.state = init_state

    # FIGURE OUT NUMPY RANDOM SEED!

    #torch.manual_seed(trial)
    #torch.use_deterministic_algorithms(True)

    map = ClutteredMap(
        Q_running_state_cost_weights,
        Phi_terminal_state_cost_weights,
        action_cost_weights,
        init_state=init_state,
        goal_state=goal_state,
        walls=walls,
        map_data=map_data,
        brt_file=brt_file,
        brt_value_threshold=brt_value_threshold,
        cost_type=cost_type)

    controller = MPPI(
        system.next_states,
        map.get_state_progress_and_obstacle_costs,
        n_states,
        num_samples=N_MPPI_SAMPLES,
        horizon=horizon,
        noise_mu=np.array(angvel_noise_mean),
        noise_sigma=np.array([angvel_noise_stddev**2]),
        u_min=np.array([angvel_min]),
        u_max=np.array([angvel_max]),
        # U_init = torch.zeros(MPPI_HORIZON, n_inputs),
        terminal_state_cost=map.get_terminal_state_cost,
        noise_abs_cost=True,
        lambda_=mppi_temperature,
        filter_nom_traj=filter_nom_traj,
        filter_samples=filter_samples,
        brt_safety_query=map.check_brt_collision,
        brt_opt_ctrl_query=map.get_brt_safety_control,
        brt_value_query=map.get_brt_value,
        brt_theta_deriv_query=map.get_brt_theta_deriv,
        diagnostics=False)  # enable_mppi_value_diagnostics

    # --- RUN EXPERIMENT ---
    # profiler.enable()
    expr_data = run_trial(system, map, controller, max_timesteps,
                          safety_filter, save_samples=False, diagnostics=False)
    # profiler.disable()
    # profiler.dump_stats(f'profile_results/vanilla_{N_MPPI_SAMPLES}_samp_nonrefactor.prof')
    # profiler.print_stats()

    print(f"Done")
