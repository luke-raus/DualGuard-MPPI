import torch
#import numpy as np
import pickle
from scipy.io import loadmat
#from scipy.interpolate import RegularGridInterpolator as SciPyRGI
from torch_interpolations import RegularGridInterpolator as TorchRGI


class ClutteredMap:
    def __init__(self,
                 running_state_cost_weights,
                 terminal_state_cost_weights,
                 action_cost_weights,
                 init_state,
                 goal_state,
                 walls,
                 map_data=None,
                 map_pickle_file=None,
                 brt_file=None,
                 brt_value_threshold=0,
                 cost_type='obs',
                 device='cpu'):

        self.d = device

        self.walls = walls   # Walls is either None or a scalar denoting +/- x/y location of walls

        self.progress_cost_weights       = running_state_cost_weights.to(device=self.d)
        self.terminal_state_cost_weights = terminal_state_cost_weights.to(device=self.d)

        self.action_cost_weights = action_cost_weights

        self.collision_cost = 1.e4

        self.goal_reward_dist = 0.1   # m
        self.goal_reward_cost = -1.e3

        self.init_state = init_state    # Should assert these are tensors of shape (3,)
        self.goal_state = goal_state.to(device=self.d)

        self.cost_type = cost_type
        assert (self.cost_type=='obs') or (self.cost_type=='brt')

        if (map_data is None):
            # Load map_data dict from pickle if not provided
            map_data = pickle.load( open( map_pickle_file, "rb" ) )
        self.obs_x = map_data['obs_x'].to(device=self.d)
        self.obs_y = map_data['obs_y'].to(device=self.d)
        self.obs_r = map_data['obs_r'].to(device=self.d)
        self.num_obstacles = len(self.obs_x)

        # --- Load BRT ---

        brt_mat = loadmat(brt_file, simplify_cells=True)

        self.brt_grid_axes = [ torch.tensor(axis).to(device=self.d, dtype=torch.float32) for axis in brt_mat['grid_axes'] ]
        #self.brt_grid_axes = tuple( brt_mat['grid_axes'] )                                      # with scipy

        self.brt_value       = torch.tensor(brt_mat['value']      ).to(device=self.d, dtype=torch.float32)
        self.brt_theta_deriv = torch.tensor(brt_mat['theta_deriv']).to(device=self.d, dtype=torch.float32)
        # Instead of storing pre-computed optimal controls over state grid, we interpolate the derivative of the 
        # value with respect to theta (theta_deriv) and choose optimal control from this 
        #self.brt_opt_ctrl  = torch.tensor(brt_mat['opt_ctrl']).to(device=self.d, dtype=torch.float32)

        """
        self.brt_grid_max_ind = torch.tensor([ len(axis)-1 for axis in self.brt_grid_axes ]).to(device=self.d, dtype=torch.int32)
        self.brt_grid_min_ind = torch.zeros_like(self.brt_grid_max_ind)

        # We're assuming that grid points are evenly spaced along each axis
        self.brt_grid_min     = torch.tensor([ axis.min()  for axis in self.brt_grid_axes ]).to(device=self.d)
        self.brt_grid_freq    = torch.tensor([ 1./(axis[1]-axis[0]) for axis in self.brt_grid_axes ]).to(device=self.d)
        """

        self.brt_value_interp = TorchRGI(self.brt_grid_axes, self.brt_value)
        #self.brt_value_interp = SciPyRGI(self.brt_grid_axes, self.brt_value, bounds_error=False, fill_value=1e4)

        self.brt_theta_deriv_interp = TorchRGI(self.brt_grid_axes, self.brt_theta_deriv)
        #self.brt_opt_ctrl_interp = TorchRGI(self.brt_grid_axes, self.brt_opt_ctrl)
        #self.brt_opt_ctrl_interp = SciPyRGI(self.brt_grid_axes, self.brt_opt_ctrl, bounds_error=False, method='nearest')

        self.brt_value_threshold = brt_value_threshold

        # See check_collision() for motivation of these variables
        self.obs_xy_unsqueezed = torch.stack((self.obs_x, self.obs_y), dim=1).unsqueeze(0)
        self.obs_r_sq_unsqueezed = self.obs_r.unsqueeze(0)**2


    def get_control_costs(self, controls):
        return (self.action_cost_weights * controls**2).sum(dim=1)

    def get_state_progress_and_obstacle_costs(self, states, controls):
        """
        states (K, nx=3), controls (K, nu=1) -> costs (K,)
        This computes the quadratic form (with diagonal weight matrix) described in SC-MPPI paper
        """

        state_diffs_sq = (states - self.goal_state)**2                       # (K, nx)
        costs_per_state_dim = state_diffs_sq * self.progress_cost_weights    # (K, nx)
        costs = costs_per_state_dim.sum(dim=1)                               # (K,)

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
        if term_states.dim() == 3:
            term_states = term_states[:,-1,:]

        state_diffs_sq = (term_states - self.goal_state)**2                      # (K, nx)
        costs_per_state_dim = state_diffs_sq * self.terminal_state_cost_weights  # (K, nx)
        costs = costs_per_state_dim.sum(dim=1)                                   # (K,)

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
        This code using broadcasting to avoid for-loop over number of obstacles
        N = number of obstacle states (in this case, 2)
        B = number of obstacles
        """

        states_expanded = states[:,0:2].unsqueeze(1)    # (K, 1, N)
        # self.obs_xy_unsqueezed...                     # (1, B, N)

        squared_diff = (states_expanded - self.obs_xy_unsqueezed)**2   # (K, B, N)
        squared_dist = squared_diff.sum(dim=-1)                        # (K, B)

        state_inside_each_obs = squared_dist <= self.obs_r_sq_unsqueezed    # (K, B)
        state_collided = state_inside_each_obs.any(dim=1)                   # (K,)

        if self.walls is not None:
            hit_wall = torch.logical_or( torch.abs(states[:,0])>self.walls,  torch.abs(states[:,1])>self.walls )
            state_collided = torch.logical_or( hit_wall, state_collided )
        # assert any_inside_obs.shape == (states.shape[0],)

        return state_collided


    def get_brt_value(self, states):
        """
        states (K, nx=3) -> BRT value (K,)
        """
        return self.brt_value_interp( states.t().contiguous() )  # with torch_interpolations


    def check_brt_collision(self, states):
        """
        states (K, nx=3) -> collision boolean (K,)
        """
        #values = torch.tensor(self.brt_value_interp(states))      # with scipy

        # scipy implementation interpolates along last (second) axis of input, whereas the 
        # unofficial torch version does so along first axis of input, hence the transpose
        # see: https://github.com/sbarratt/torch_interpolations/issues/1

        return self.get_brt_value( states ) <= self.brt_value_threshold

    def get_brt_theta_deriv(self, states):
        return self.brt_theta_deriv_interp( states.t().contiguous() )


    def brt_opt_ctrl(self, states, max_angvel=6.0):
        """
        states (K, nx=3) -> control (K, nu=1)
        """
        theta_deriv = self.get_brt_theta_deriv( states )

        # In MATLAB:  uOpt = (deriv{3}>=0)*obj.wRange(2) + (deriv{3}<0)*(obj.wRange(1));
        opt_ctrl = (theta_deriv >= 0)*(max_angvel) + (theta_deriv < 0)*(-max_angvel)
        return opt_ctrl


    def get_brt_safety_control(self, states):#, linvel=4.0):
        """
        states (K, nx=3) -> control (K, nu=2)
        """
        controls = self.brt_opt_ctrl( states ).unsqueeze(dim=1)         # (K,nu=1)
        return controls
