import numpy as np
from typing import Callable
# import time


class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    This implementation derived from: https://github.com/ferreirafabio/mppi_pendulum
    which was implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning'
    """

    def __init__(self,
        dynamics:      Callable,
        running_cost:  Callable,
        state_dim:     int,
        num_samples:   int,
        horizon:       int,
        u_min:         np.ndarray,
        u_max:         np.ndarray,
        u_noise_mean:  np.ndarray,
        u_noise_covar: np.ndarray,
        terminal_state_cost:   Callable = None,
        temperature:           float = 1.,
        u_init:                np.ndarray = None,
        U_init_is_random:      bool = True,
        u_per_command:         int = 1,
        sample_null_action:    bool = False,
        noise_abs_cost:        bool = True,
        filter_samples:        bool = False,
        filter_nom_traj:       bool = False,
        brt_safety_query:      Callable = None,
        brt_opt_ctrl_query:    Callable = None,
        brt_value_query:       Callable = None,
        brt_theta_deriv_query: Callable = None,
        random_seed:           int = None,
        diagnostics:           bool = False,
    ):
        """
        :param dynamics: function(state, action) -> next_state (K x state_dim) taking in batch state (K x state_dim) and action (K x control_dim)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param state_dim: state dimension
        :param u_noise_covar: (control_dim x control_dim) control noise covariance (assume v_t ~ N(u_t, u_noise_covar))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param temperature: temperature, positive scalar where larger values will allow more exploration
        :param u_noise_mean: (control_dim) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (control_dim) minimum values for each dimension of control to pass into dynamics
        :param u_max: (control_dim) maximum values for each dimension of control to pass into dynamics
        :param u_init: (control_dim) what to initialize new end of trajectory control to be; defaults to zero
        :param U_init_is_random: [bool] is U_init noise? Otherwise, it's a constant sequence of mean control
        #:param U_init: (T x control_dim) initial control sequence; defaults to noise
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.K = num_samples  # N_SAMPLES
        self.T = horizon      # TIMESTEPS

        # If u_noise_mean is scalar, we turn it into 1-d numpy array
        self.u_noise_mean = u_noise_mean
        if isinstance(u_noise_mean, (float, int)):
            self.u_noise_mean = np.array([u_noise_mean])

        # If u_noise_covar is scalar, we turn it into 2-d numpy array
        self.u_noise_covar = u_noise_covar
        if isinstance(self.u_noise_covar, (float, int)):
            self.u_noise_covar = np.array([[u_noise_covar]])            


        # dimensions of state and control
        self.state_dim = state_dim
        self.control_dim = 1 if len(self.u_noise_covar.shape) == 0 else self.u_noise_covar.shape[0]
        self.temperature = temperature

        # handle 1D edge case
        if self.control_dim == 1:
            u_noise_mean = np.array([u_noise_mean])
            u_noise_covar = np.array([u_noise_covar])

        if u_noise_mean is None:
            u_noise_mean = np.zeros(self.control_dim)

        if u_init is None:
            u_init = np.zeros_like(u_noise_mean)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not isinstance(self.u_max, np.ndarray):
                self.u_max = np.array(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not isinstance(self.u_min, np.ndarray):
                self.u_min = np.array(self.u_min)
            self.u_max = -self.u_min

        if self.u_noise_covar.ndim == 1:
            self.u_noise_covar_inv = np.linalg.inv(np.expand_dims(self.u_noise_covar, axis=0))
        else:
            self.u_noise_covar_inv = np.linalg.inv( self.u_noise_covar )

        self.numpy_rand_gen = np.random.default_rng(seed=random_seed)

        # T x control_dim control sequence
        self.u_init = u_init

        if U_init_is_random:
            # TODO Need u_noise_mean to be 1-dimensional & u_noise_covar to be 2-dimensional
            # This works for now because u_noise_mean is scalar & u_noise_covar is 1d
            some_noise = self.numpy_rand_gen.multivariate_normal(self.u_noise_mean, self.u_noise_covar, size=(self.T) )
            self.U = self._bound_action( some_noise )
        else:
            self.U = np.full((self.T, self.control_dim), self.u_noise_mean)   # [T * control_dim]


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
        # if self.filter_samples or self.filter_nom_traj:
        self.brt_safety_query = brt_safety_query
        self.brt_opt_ctrl_query = brt_opt_ctrl_query

        self.brt_value_query = brt_value_query
        self.brt_theta_deriv_query = brt_theta_deriv_query
        
        # Need to do this so we can have nominal_trajectory right when initialized
        self.nominal_trajectory = self._get_nominal_trajectory()

    def _dynamics(self, state, u, t):
        return self.F(state, u)

    def _running_cost(self, state, u, t):
        return self.running_cost(state, u)


    def reset_controller(self):
        pass


    

    def command(self, state, shift_nominal_trajectory=True):
        """
        :param state: (state_dim) or (K x state_dim) current state, or samples of states (for propagating a distribution of states)
        :param shift_nominal_trajectory: Whether to roll the nominal trajectory forward one step. This should be True
        if the command is to be executed. If the nominal trajectory is to be refined then it should be False.
        :returns action: (control_dim) best action
        """
        if shift_nominal_trajectory:
            # shift command 1 time step
            self.U = np.roll(self.U, -1, axis=0)
            self.U[-1] = self.u_init
            assert self.U.shape == (self.T, 1)

        self.state = state
        cost_total = self._compute_total_cost_batch()
        beta = np.min(cost_total)
        self.cost_total_non_zero = np.exp( (-1/self.temperature) * (cost_total - beta) )
        eta = np.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        perturbations = []
        for t in range(self.T):
            perturbations.append(np.sum(np.expand_dims(self.omega, axis=1) * self.noise[:, t], axis=0))
        perturbations = np.stack(perturbations)
        #print(perturbations.shape)
        self.U = self.U + perturbations
        assert self.U.shape == (self.T, 1)

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
        K, T, control_dim = self.perturbed_action.shape
        #print(f"{K}, {T}, {nu}, self: {self.nu}")
        assert control_dim == self.control_dim

        cost_total = np.zeros(K)

        if self.state.shape == (K, self.state_dim):
            state = self.state
        else:
            state = np.tile(self.state, (K, 1))

        sampled_states = [state]
        sampled_actions = []
        sample_brt_values = []
        sample_safety_filter = []
        sample_brt_theta_deriv = []

        has_reached_goal = np.zeros_like(cost_total)
        for t in range(T):
            u = self.perturbed_action[:, t]
            if self.filter_samples:

                # --- SAFETY FILTERING (Pre-emptive) ---
                # Try dynamics with preliminarily-filtered controls
                potential_next_state = self._dynamics(state, u, t)

                # If this control takes system into unsafe state, apply safety controller preemptively on these samples
                next_state_is_unsafe = self.brt_safety_query(potential_next_state)
                u[next_state_is_unsafe,:] = self.brt_opt_ctrl_query(state[next_state_is_unsafe,:])

                potential_next_state[next_state_is_unsafe,:] = self._dynamics(state[next_state_is_unsafe,:], u[next_state_is_unsafe,:], t)
                # NOTE! Had a nasty bug where the above 2 lines modified `state` directly`
                # This meant we never had state set directly to a fresh array, and were instead updated in-place
                # which meant that we were effectively appending *references* of state to the list, all of which changed when we changed state
                # Hence the stacked array had the same state for every timestep. SMH!
                state = potential_next_state

                sample_brt_values.append( self.brt_value_query(state) )
                sample_safety_filter.append( next_state_is_unsafe )
                sample_brt_theta_deriv.append( self.brt_theta_deriv_query(state) )
            else:
                # DYNAMICS
                state = self._dynamics(state, u, t)

            c, is_in_goal = self._running_cost(state, u, t)

            # Only accumulate cost for samples which have not reached goal
            cost_total[np.logical_not(has_reached_goal)] += c[np.logical_not(has_reached_goal)]

            # Need to allow cost from one step to accumulate first 
            has_reached_goal = np.logical_or(has_reached_goal, is_in_goal)

            # Save total states/actions
            sampled_actions.append(u)
            sampled_states.append(state)

        # Actions is:  K x T x control_dim
        # States is:   K x T x state_dim
        sampled_actions = np.stack(sampled_actions, axis=-2)
        sampled_states  = np.stack(sampled_states,  axis=-2)    # NOTE: States is actually K,(T+1),state_dim 

        if self.filter_samples:
            # These are:     K x T x 1
            self.sample_brt_values       = np.concatenate(sample_brt_values,      axis=-1)
            self.sample_safety_filter    = np.concatenate(sample_safety_filter,   axis=-1)
            self.sample_brt_theta_deriv  = np.concatenate(sample_brt_theta_deriv, axis=-1)
        else:
            self.sample_brt_values       = []
            self.sample_safety_filter    = []
            self.sample_brt_theta_deriv  = []

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(sampled_states, sampled_actions)
            cost_total = cost_total + c

        return cost_total, sampled_states, sampled_actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.numpy_rand_gen.multivariate_normal(self.u_noise_mean, self.u_noise_covar, size=(self.K, self.T) )
        assert noise.shape == (self.K, self.T, 1)
        # broadcast own control to noise over samples; now it's K x T x control_dim
        #print(self.U.shape)
        #print(noise.shape)
        perturbed_action = np.expand_dims(self.U, axis=0) + noise
        assert perturbed_action.shape == (self.K, self.T, 1)
        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)
        assert self.perturbed_action.shape == (self.K, self.T, 1)

        # This function may also update self.perturbed_action via safety filtering
        rollout_cost, self.sampled_states, self.sampled_actions = self._compute_rollout_costs()

        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        #self.noise = self.perturbed_action - self.U
        self.noise = self.sampled_actions - self.U
        assert self.noise.shape == (self.K, self.T, 1)
        if self.noise_abs_cost:
            action_cost = self.temperature * np.abs(self.noise) @ self.u_noise_covar_inv
            # NOTE: The original paper does self.temperature * torch.abs(self.noise) @ self.u_noise_covar_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.temperature * self.noise @ self.u_noise_covar_inv  # Like original paper

        # action perturbation cost
        perturbation_cost = np.sum(self.U * action_cost, axis=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = np.maximum(np.minimum(u, self.u_max), self.u_min)
                action[:, self._slice_control(t)] = cu
        return action

    def _filter_nominal_trajectory(self):
        pass

    def _get_nominal_trajectory(self):
        """
        Returns (T+1 x state_dim) nominal state trajectory based on current state
          (self.state) and nominal control plan (self.U), including current state
        """
        states = np.zeros((self.T + 1, self.state_dim))
        states[0,:] = self.state
        #print(f"nom traj state: {states[0,:].size()}, u: {self.U[0,:].size()}") 

        for i in range(self.T):
            states[i+1,:] = self._dynamics(states[i,:], self.U[i,:], i)
        return states

    def _slice_control(self, t):
        return slice(t * self.control_dim, (t + 1) * self.control_dim)

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            # truncate trajectory
            self.U = self.U[:horizon]
        elif horizon > self.U.shape[0]:
            # extend with u_init
            self.U = np.concatenate((self.U, self.u_init.repeat(horizon - self.U.shape[0], 1)))
        self.T = horizon

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.numpy_rand_gen.multivariate_normal(self.u_noise_mean, self.u_noise_covar, size=(self.T) )
        #self.U = self.noise_dist.sample((self.T,))
