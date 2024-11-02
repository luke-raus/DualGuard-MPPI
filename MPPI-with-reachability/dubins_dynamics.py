import numpy as np


class DubinsCarFixedVel():
    """
    Using the 3-state, 2-input Dubins dynamics described in section VI-B of
    Gandhi et al 2023 on Safe Importance Sampling in MPPI
    """

    def __init__(self, timestep, linvel, init_state):
        """
        Initializes a class of the SimpleDubins vehicle, with a 'nominal' motion model
        """

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
        #TODO: Reference for exact Dubins integration

        states (K, nx=3), controls (K, nu=1) -> next_states (K, nx=3)

        For duration of timestep, system travels in arc of curvature/radius:
            K = angvel/linvel
            r = linvel/angvel
        Where amount turned is difference in heading:
            d_theta = angvel*dt
        Center point C=(C_x;C_y) of arc based on current state (x;y;heading) & radius:
            C_x = S_x - r*sin(theta)
            C_y = S_y + r*cos(theta)
        Now we get the new state x;y vector S' by this procedure:
            Get vector P of length r from rotation center C to state S (S=C+P -> P=S-C)
            Rotate P by d_theta rotation matrix to get P'
            Add P' back to C to get S' (S'=C+P')
        Expanding P', this looks like:
            S' = C + R_{d_theta}@(S-C)
        """
        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        angvel = controls[:, 0]

        # Get indices of zero-inputs for regular straight-line integration
        no_steer = (angvel == 0.0)

        # FIXME: This dubins integration seems overly complex for a fixed-velocity model we should use the usual form which should be faster an has no weird edge cases   
        # check discrete_dynamics() in experiment_runner.py for a simpler version     
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
        Option single state   - inputs are dim=1 tensors: state (nx=3), control (nu=1)
        Option multiple sttae - inputs are dim=2 tensors: state (K, nx=3), control (K, nu=1)
        """
        if state.ndim == 1:
            # Singleton state input; unsqueeze to pass to batch function
            assert (control.ndim == 1)
            return self._next_states_batch(np.expand_dims(state, axis=0), np.expand_dims(control, axis=0)).squeeze(axis=0)
        else:
            assert (control.ndim > 1)
            return self._next_states_batch(state, control)

    def update_true_state(self, control):
        """
        Apply single control input to update state of system
        Input:
            control is (nu)
        Output:
            None; just updates current true state (self.state)
        """
        self.state = self.next_states(self.state, control)
