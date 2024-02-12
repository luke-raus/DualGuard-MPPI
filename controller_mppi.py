import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=15, device="cpu", #"cuda",
                terminal_state_cost=None,
                lambda_=1.,
                noise_mu=None,
                u_min=None,
                u_max=None,
                u_init=None,
                U_init=None,
                u_per_command=1,
                sample_null_action=False,
                noise_abs_cost=True,
                filter_samples=False,
                filter_nom_traj=False,
                brt_safety_query=None,
                brt_opt_ctrl_query=None,
                brt_value_query=None,
                brt_theta_deriv_query=None,
                diagnostics=False):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon      # TIMESTEPS

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)

        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self._bound_action(self.noise_dist.sample((self.T,)))

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None

        self.filter_samples = filter_samples
        self.filter_nom_traj = filter_nom_traj
        if self.filter_samples or self.filter_nom_traj:
            self.brt_safety_query = brt_safety_query
            self.brt_opt_ctrl_query = brt_opt_ctrl_query
        
        self.diagnostics = diagnostics
        if self.diagnostics:
            self.brt_value_query = brt_value_query
            self.brt_theta_deriv_query = brt_theta_deriv_query

    def _dynamics(self, state, u, t):
        return self.F(state, u)

    def _running_cost(self, state, u, t):
        return self.running_cost(state, u)

    def command(self, state, shift_nominal_trajectory=True):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :param shift_nominal_trajectory: Whether to roll the nominal trajectory forward one step. This should be True
        if the command is to be executed. If the nominal trajectory is to be refined then it should be False.
        :returns action: (nu) best action
        """
        if shift_nominal_trajectory:
            # shift command 1 time step
            self.U = torch.roll(self.U, -1, dims=0)
            self.U[-1] = self.u_init

        return self._command(state)

    def _command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()
        beta = torch.min(cost_total)
        self.cost_total_non_zero = torch.exp( (-1/self.lambda_) * (cost_total - beta) )
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        perturbations = []
        for t in range(self.T):
            perturbations.append(torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0))
        perturbations = torch.stack(perturbations)
        self.U = self.U + perturbations
        self.nominal_trajectory = self._get_nominal_trajectory()

        # NOTE: Probably makes more sense to filter nominal trajectory at end (?)

        # Filter nominal trajectory here...
        #self._filter_nominal_trajectory()
        # This updates self.U and self.nominal_trajectory

        action = self.U[:self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]
        
        return action

    def _compute_rollout_costs(self):
        K, T, nu = self.perturbed_action.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)

        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        sampled_states = [state]
        sampled_actions = []
        if self.diagnostics and self.filter_samples:
            sample_brt_values = []
            sample_safety_filter = []
            sample_brt_theta_deriv = []

        has_reached_goal = torch.zeros_like(cost_total)
        for t in range(T):
            u = self.perturbed_action[:, t]
            if self.filter_samples:

                # --- SAFETY FILTERING (Pre-emptive) ---
                # Try dynamics with preliminarily-filtered controls
                potential_next_state = self._dynamics(state, u, t)
                # If this control takes system into unsafe state, apply safety controller preemptively
                next_state_is_unsafe = self.brt_safety_query(potential_next_state)
                u[next_state_is_unsafe,:] = self.brt_opt_ctrl_query(state[next_state_is_unsafe,:])

                state[~next_state_is_unsafe,:] = potential_next_state[~next_state_is_unsafe,:]
                state[ next_state_is_unsafe,:] = self._dynamics(state[ next_state_is_unsafe,:], u[next_state_is_unsafe,:], t)

                if self.diagnostics:
                    sample_brt_values.append( self.brt_value_query(state) )
                    sample_safety_filter.append( next_state_is_unsafe )
                    sample_brt_theta_deriv.append( self.brt_theta_deriv_query(state) )
            else:
                # Apply potentially-filtered controls to get next state
                state = self._dynamics(state, u, t)

            c, is_in_goal = self._running_cost(state, u, t)

            # Only accumulate cost for samples which have not reached goal
            cost_total[torch.logical_not(has_reached_goal)] += c[torch.logical_not(has_reached_goal)]

            # Need to allow cost from one step to accumulate first 
            has_reached_goal = torch.logical_or(has_reached_goal, is_in_goal)

            # Save total states/actions
            sampled_actions.append(u)
            sampled_states.append(state)
  
        # Actions is:  K x T x nu
        # States is:   K x T x nx
        sampled_actions = torch.stack(sampled_actions, dim=-2)
        sampled_states  = torch.stack(sampled_states,  dim=-2)
        if self.diagnostics:
            # These are:     K x T x 1
            self.sample_brt_values       = torch.stack(sample_brt_values,      dim=-1)
            self.sample_safety_filter    = torch.stack(sample_safety_filter,   dim=-1)
            self.sample_brt_theta_deriv  = torch.stack(sample_brt_theta_deriv, dim=-1)


        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(sampled_states, sampled_actions)
            cost_total = cost_total + c
        return cost_total, sampled_states, sampled_actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.noise_dist.rsample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + noise
        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)

        # This function may also update self.perturbed_action via safety filtering
        rollout_cost, self.sampled_states, self.sampled_actions = self._compute_rollout_costs()

        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        #self.noise = self.perturbed_action - self.U
        self.noise = self.sampled_actions - self.U
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv  # Like original paper

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = torch.max(torch.min(u, self.u_max), self.u_min)
                action[:, self._slice_control(t)] = cu
        return action

    def _filter_nominal_trajectory(self):
        pass

    def _get_nominal_trajectory(self):
        """
        Returns (T+1 x nx) nominal state trajectory based on current state
          (self.state) and nominal control plan (self.U), including current state
        """
        states = torch.zeros(self.T + 1, self.nx).to(device=self.d)
        states[0,:] = self.state
        for i in range(self.T):
            states[i+1,:] = self._dynamics(states[i,:], self.U[i,:], i)
        return states

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            # truncate trajectory
            self.U = self.U[:horizon]
        elif horizon > self.U.shape[0]:
            # extend with u_init
            self.U = torch.cat((self.U, self.u_init.repeat(horizon - self.U.shape[0], 1)))
        self.T = horizon

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))
