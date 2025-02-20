import numpy as np
from scipy.optimize import minimize

# nonlinear predictive filter
def control_shielding(u, q0, N, dt, V, beta, map):
    # Parameters
    # V: Speed of the car
    # N: horizon that control shielding taken into account
    # dt: Time discretization step

    def discrete_dynamics(q, w, V):
        x, y, theta= q
        next_q = np.zeros(3)
        next_q[0] = x + dt * V * np.cos(theta)
        next_q[1] = y + dt * V * np.sin(theta)
        next_q[2] = theta + dt * w
        next_q[2] = (next_q[2] + np.pi) % (2 * np.pi) - np.pi
        return next_q

    # Objective function
    def control_shielding_objective(U, q_init, N, V, beta, map):
        J = 0.0
        q = q_init.copy()  # Initial state
        for k in range(N):
            w = U[k]
            next_q = discrete_dynamics(q, w, V)
            J += min( map.get_brt_value(next_q) - (1-beta) * map.get_brt_value(q), 0)
            q = next_q
        return -J  # Negate to convert maximization to minimization

    # reshape control input to a 1D sequence for minimize method
    # reshaped_u = u[:N, ...].copy()
    # reshaped_u = reshaped_u.reshape(np.prod(reshaped_u.shape))

    # Run optimization, passing the initial state as an argument
    result = minimize(control_shielding_objective, u.squeeze(), args=(q0, N, V, beta, map), method='BFGS', options={"maxiter": 10})

    # Extract optimized control sequence
    opt_U = result.x
    return np.array([opt_U[0]])