from controller_mppi import MPPI
from environment_cluttered_goal_reward import ClutteredMap
from dubins_dynamics import DubinsCarFixedVel
from run_trial import run_trial


import torch
import pickle


if __name__ == "__main__":

    device = 'cpu'

    experiment_time_horizon = 15   # sec

    timestep = 0.02
    max_timesteps = round(experiment_time_horizon / timestep)

    N_MPPI_SAMPLES = 1000

    # Dyn. sys. definition
    angvel_min, angvel_max = (-6.0, 6.0)   # rad/s
    angvel_noise_mean = 0.0
    linvel = 4.0           # m/s

    angvel_noise_stddev = 8.0

    # unused currently...
    Phi_terminal_state_cost_weights = torch.tensor([0., 0., 0.])
    Q_running_state_cost_weights = torch.tensor([1., 1., 0.])
    action_cost_weights = 0.2

    state_pairs = pickle.load(open("state_pairs_outside_disturbed_brt_dec_29.pkl", "rb"))

    map_data = pickle.load(open("new_map_aug_12.pkl", "rb"))
    walls = 5.0

    brt_file = "brt_dubin_new_map_disturbed_aug_16_highres.mat"
    brt_value_threshold = 0.01

    placeholder_init_state = torch.tensor([0., 0., 0.])

    system = DubinsCarFixedVel(timestep,
                               linvel,
                               placeholder_init_state,
                               device=device)
    n_states = system.nx
    n_inputs = system.nu

    horizon = 25
    mppi_temperature = 5.0

    for trial in range(1):
        for config in [5]:

            if config == 0:
                plot_title = "Vanilla MPPI with obstacle costs"
                filter_samples, safety_filter, cost_type = False, False, 'obs'
            elif config == 1:
                plot_title = "Vanilla MPPI with obstacle costs + safety filter"
                filter_samples, safety_filter, cost_type = False, True, 'obs'
            elif config == 2:
                plot_title = "Vanilla MPPI with BRT costs"
                filter_samples, safety_filter, cost_type = False, False, 'brt'
            elif config == 3:
                plot_title = "Vanilla MPPI with BRT costs + safety filter"
                filter_samples, safety_filter, cost_type = False, True, 'brt'
            elif config == 4:
                plot_title = "Our method, obstacles in cost"
                filter_samples, safety_filter, cost_type = True, False, 'obs'

            filter_nom_traj = False

            init_state = state_pairs['init'][trial].to(device=device)
            goal_state = state_pairs['goal'][trial].to(device=device)

            system.state = init_state.to(device=device)

            torch.manual_seed(trial)
            torch.use_deterministic_algorithms(True)

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
                cost_type=cost_type,
                device=device)

            controller = MPPI(
                system.next_states,
                map.get_state_progress_and_obstacle_costs,
                n_states,
                num_samples=N_MPPI_SAMPLES,
                horizon=horizon,
                noise_mu=torch.tensor(angvel_noise_mean),
                noise_sigma=torch.tensor([angvel_noise_stddev**2]),
                u_min=torch.tensor([angvel_min]),
                u_max=torch.tensor([angvel_max]),
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
                diagnostics=False,  # enable_mppi_value_diagnostics,
                device=device)

            # --- RUN EXPERIMENT ---
            expr_data = run_trial(system, map, controller, max_timesteps,
                                  safety_filter, save_samples=False, diagnostics=False)

            pickle.dump(expr_data, open(f"results_trial_{trial}_config_{config}.pkl", "wb"))

            print(f"Done with trial {trial}, config {config}")
