import numpy as np


class DubinsCarFixedVel():
    """
    Implements 3-state, 1-input (steering) dynamics of fixed-speed Dubins Car
    """

    def __init__(self, timestep, linvel, init_state):
        self.nx = 3
        self.nu = 1
        self.linvel = linvel
        self.state = init_state
        self.timestep = timestep

    def get_state(self):
        return self.state

    def set_state(self, x, y, theta):
        self.state[0] = x
        self.state[1] = y
        self.state[2] = theta

    def _next_states_batch(self, states, controls):
        """
        Performs exact forward-integration of Dubins dynamics

        states (K, nx=3), controls (K, nu=1) -> next_states (K, nx=3)
        """
        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        angvel = controls[:, 0]

        # Get indices of zero-inputs for regular straight-line integration
        no_steer = (angvel == 0.0)
  
        # Compute turning radius -> steering center -> new state x/y as above
        # This won't fail if omega=0: will just produce NaN/Inf's
        d_theta = self.timestep * angvel
        radius = self.linvel / angvel

        # Angle sum trig identities applied to expansion
        x_next = x + radius * (-np.sin(theta) + np.sin(theta + d_theta))
        y_next = y + radius * (np.cos(theta) - np.cos(theta + d_theta))

        # Apply regular Euler integration if not steering, replacing NaN/Inf's
        x_next[no_steer] = x[no_steer] + self.linvel * self.timestep * np.cos(theta[no_steer])
        y_next[no_steer] = y[no_steer] + self.linvel * self.timestep * np.sin(theta[no_steer])

        # Add theta & wrap around at 2*pi
        theta_next = theta + d_theta
        theta_next = ((theta_next + np.pi) % (2 * np.pi)) - np.pi

        # Stack along dimension 1 to go from 3 vectors of K values to K (state) vectors of 3 values
        return np.column_stack((x_next, y_next, theta_next))

    def next_states(self, state, control):
        """
        state, control -> next_state
        Option single state   - inputs are dim=1 arrays: state (nx=3), control (nu=1)
        Option multiple sttae - inputs are dim=2 arrays: state (K, nx=3), control (K, nu=1)
        """
        if state.ndim == 1:
            # Singleton state input; unsqueeze to pass to batch function
            assert (control.ndim == 1)
            return self._next_states_batch(np.expand_dims(state, axis=0), np.expand_dims(control, axis=0)).squeeze(axis=0)
        else:
            assert (control.ndim > 1)
            return self._next_states_batch(state, control)

    def update_true_state(self, control) -> None:
        """
        Apply single control input to update state of system
        """
        self.state = self.next_states(self.state, control)
