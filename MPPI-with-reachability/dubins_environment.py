import numpy as np
import h5py
# from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator as SciPyRGI


class ClutteredMap:
    def __init__(self,
                 running_state_cost_weights,
                 terminal_state_cost_weights,
                 action_cost_weights,
                 init_state,
                 goal_state,
                 brt_fname,
                 brt_value_threshold=0,
                 cost_type='obs'):

        self.progress_cost_weights       = running_state_cost_weights
        self.terminal_state_cost_weights = terminal_state_cost_weights

        self.action_cost_weights = action_cost_weights

        self.collision_cost = 1.e4

        self.goal_reward_dist = 0.1   # m
        self.goal_reward_cost = -1.e3

        self.init_state = init_state    # Should assert these are tensors of shape (3,)
        self.goal_state = goal_state

        self.cost_type = cost_type
        assert (self.cost_type=='obs') or (self.cost_type=='brt')

        # --- Load BRT from .hdf5 file ---

        with h5py.File(brt_fname, 'r') as hdf_file:
            # Load the arrays from the file
            self.obs_value       = hdf_file['obstacle_value_grid'][:]
            self.brt_value       = hdf_file['brt_value_grid'][:]
            self.brt_theta_deriv = hdf_file['brt_theta_deriv_grid'][:]

            self.brt_grid_axes = tuple([ hdf_file['brt_axes'][f'axis_{i}'][:] for i in range(self.brt_value.ndim) ])

        self.brt_obs_interp         = SciPyRGI(self.brt_grid_axes, self.obs_value, bounds_error=False)
        self.brt_value_interp       = SciPyRGI(self.brt_grid_axes, self.brt_value, bounds_error=False)
        self.brt_theta_deriv_interp = SciPyRGI(self.brt_grid_axes, self.brt_theta_deriv)

        self.brt_value_threshold = brt_value_threshold


    def get_control_costs(self, controls):
        return (self.action_cost_weights * controls**2).sum(axis=1)

    def get_state_progress_and_obstacle_costs(self, states, controls):
        """
        states (K, nx=3), controls (K, nu=1) -> costs (K,)
        This computes the quadratic form (with diagonal weight matrix) described in SC-MPPI paper
        """

        state_diffs_sq = (states - self.goal_state)**2                       # (K, nx)
        costs_per_state_dim = state_diffs_sq * self.progress_cost_weights    # (K, nx)
        costs = costs_per_state_dim.sum(axis=1)                              # (K,)

        is_in_goal_reward_dist = ( (state_diffs_sq[:,0] + state_diffs_sq[:,1])  <= self.goal_reward_dist**2 )   # bool (K,)
        costs[is_in_goal_reward_dist] = self.goal_reward_cost

        costs += self.get_control_costs(controls)

        costs += self.get_obstacle_costs(states)                     # (K,)
        return costs, is_in_goal_reward_dist


    def get_terminal_state_cost(self, term_states, controls=None):
        """
        states (K, nx=3) -> costs (K,)
        """
        # Default algorithm passes in a (K, T, nx) tensor, so we get slice of final T
        if term_states.ndim == 3:
            term_states = term_states[:,-1,:]

        state_diffs_sq = (term_states - self.goal_state)**2                      # (K, nx)
        costs_per_state_dim = state_diffs_sq * self.terminal_state_cost_weights  # (K, nx)
        costs = costs_per_state_dim.sum(axis=1)                                  # (K,)

        costs += self.get_obstacle_costs(term_states)                            # (K,)
        return costs


    def get_obstacle_costs(self, states):
        if self.cost_type == 'obs':
            return self.collision_cost * self.check_obs_collision(states)
        elif self.cost_type == 'brt':
            return self.collision_cost * self.check_brt_collision(states)
        else:
            raise('Undefined obstacle type')


    def check_obs_collision(self, states):
        """
        states (K, nx=3) -> collision boolean (K,)
        """
        return self.brt_obs_interp( states ) < 0.0


    def get_brt_value(self, states):
        """
        states (K, nx=3) -> BRT value (K,)
        """
        return self.brt_value_interp( states )


    def check_brt_collision(self, states):
        """
        states (K, nx=3) -> collision boolean (K,)
        """
        return self.get_brt_value( states ) <= self.brt_value_threshold


    def get_brt_theta_deriv(self, states):
        return self.brt_theta_deriv_interp( states )


    def brt_opt_ctrl(self, states, max_angvel=6.0):
        """
        states (K, nx=3) -> control (K, nu=1)
        """
        theta_deriv = self.get_brt_theta_deriv( states )

        # In MATLAB:  uOpt = (deriv{3}>=0)*obj.wRange(2) + (deriv{3}<0)*(obj.wRange(1));
        opt_ctrl = (theta_deriv >= 0)*(max_angvel) + (theta_deriv < 0)*(-max_angvel)
        return opt_ctrl


    def get_brt_safety_control(self, states):
        """
        states (K, nx=3) -> control (K, nu=2)
        """
        controls = np.expand_dims(self.brt_opt_ctrl( states ), axis=1)         # (K,nu=1)
        return controls
