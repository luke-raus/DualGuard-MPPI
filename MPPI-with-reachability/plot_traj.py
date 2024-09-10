import plotly.graph_objects as go
import numpy as np
import h5py
from pathlib import Path

from experiment_result import ExperimentResult


def update_plot_layout_with_map(layout:dict, environment:Path = None) -> dict:
    w   = 5.    # wall distances from center
    w_t = 1.    # wall display thickness
    walls = [[-(w+w_t),       w,      -w,       -w],
             [-(w+w_t), (w+w_t), (w+w_t),        w],
             [-(w+w_t),      -w, (w+w_t), -(w+w_t)],
             [       w,       w, (w+w_t),       -w]]

    obs_rects = [{'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'type': 'rect', 'xref': 'x', 'yref': 'y',
                'fillcolor': 'black', 'opacity': 0.5, 'line': {'width': 0}} for x0, y0, x1, y1 in walls]



    with h5py.File(environment, 'r') as f:
        x = f['obstacles']['obstacle_x'][:]
        y = f['obstacles']['obstacle_y'][:]
        r = f['obstacles']['obstacle_radius'][:]
    obs_circs = [{'x0': float(x[i]-r[i]), 'y0': float(y[i]-r[i]), 'x1': float(x[i]+r[i]), 'y1': float(y[i]+r[i]),
                    'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'black', 'opacity': 0.5, 'line': {'width': 0}} for i in range(len(x))]
    # circle_settings = {'type':'circle', 'xref':'x', 'yref':'y', 'fillcolor':'gray', 'layer':'below', 'opacity':1.0}
    # points = [ go.layout.Shape(x0=x[i]-r[i], y0=y[i]-r[i], x1=x[i]+r[i], y1=y[i]+r[i], **circle_settings) for  ]

    layout["shapes"] = tuple(obs_rects) + tuple(obs_circs)
    return layout


def get_trace_of_overall_trajectory_to_index(result:ExperimentResult, index:int = -1) -> dict:

    trajectory = result.get_overall_trajectory()
    if index == -1:
        index = np.shape(trajectory)[0]

    trajectory_trace = {
        "x": trajectory[:index, 0],
        "y": trajectory[:index, 1],
        "mode": "lines+markers",
        "showlegend": False,
        "line": dict(color='black'),
        "name": "Actual trajectory"
    }
    return trajectory_trace


def get_traces_of_samples(step_data:dict, max_samples=100) -> list:
    """
    frame["data"].extend( THIS )
    """
    sample_traces = []

    sample_states  = step_data['sample_states']   # (K, T, nx)
    sample_costs   = step_data['sample_costs']     # (K)
    sample_weights = step_data['sample_weights']   # (K)
    """
    # Compute cost min/max at timestep
    # max_weight = max(sample_weights)
    # range_weight = max_weight - min(sample_weights)
    cost_threshold = 1e4

    if np.any(sample_costs < cost_threshold):
        max_cost_below_thresh = np.max(sample_costs[sample_costs < cost_threshold])
    else:
        max_cost_below_thresh = np.max(sample_costs)
    min_cost = np.min(sample_costs)
    # to ensure nonzero...
    range_cost = max_cost_below_thresh - min_cost + 1e-8
    # Tensor to list...
    sample_costs = sample_costs.tolist()
    """
    nSamples = min(np.shape(sample_states)[0], max_samples)

    sample_colors = []
    sample_alpha = 0.4
    sample_color = f"rgba(255,140,16,{sample_alpha})"

    sample_traces = [ {
        "x": sample_states[i][:, 0],   #.tolist()
        "y": sample_states[i][:, 1],
        "mode": "lines",
        "showlegend": False,
        "line": {"color": sample_color},
        "name": f"sample {i}",
        "text": f"cost: {round(sample_costs[i], 1)}, weight: {round(sample_weights[i], 6)}",
    } for i in range(nSamples) ]

    """
    # If hit an obstacle, make orange
    if sample_cost > cost_threshold:
        sample_color = f"rgba(255,140,16,{sample_alpha})"
    else:
        # maps to 0-1, then to 0-255
        sample_cost_scaled = round(
            255*float((sample_cost - min_cost) / range_cost))
        sample_color = f"rgba({sample_cost_scaled},0,{255-sample_cost_scaled},{sample_alpha})"
    """

    return sample_traces


def get_trace_of_nominal_traj_before(step_data:dict) -> dict:
        return {
            "x": step_data["nominal_traj_states_before"][:, 0],
            "y": step_data["nominal_traj_states_before"][:, 1],
            "mode": "lines",
            "showlegend": True,
            "line": dict(color='rgba(0,255,0,0.2)', width=4),
            "name": "Nominal trajectory before"
        }


def get_trace_of_nominal_traj_after(step_data:dict) -> dict:
        return {
            "x": step_data["nominal_traj_states_after"][:, 0],
            "y": step_data["nominal_traj_states_after"][:, 1],
            "mode": "lines",
            "showlegend": True,
            "line": dict(color='rgba(0,255,0,1.0)', width=4),
            "name": "Nominal trajectory after"
        }

# def plot_experiment_at_timestep(result:ExperimentResult, environment_file:str, step_index:int) -> go.Figure:
def plot_experiment_at_timestep(result:ExperimentResult, step_index:int) -> go.Figure:

    max_samples = 200

    traces = []
    traces.append(get_trace_of_overall_trajectory_to_index(result, index=step_index))

    step_data = result.get_timestep_data(step_index)

    has_samples = result.get_config()['save_samples']
    if has_samples:
        traces.extend(get_traces_of_samples(step_data, max_samples=max_samples))

    traces.append(get_trace_of_nominal_traj_before(step_data))
    traces.append(get_trace_of_nominal_traj_after(step_data))

    layout = update_plot_layout_with_map( {}, result.get_environment_path() )

    layout["xaxis"] = {'showgrid': False, 'zeroline': False}
    layout["yaxis"] = {'showgrid': False, 'zeroline': False}

    fig_dict = { "data": traces, "layout": layout }
    fig = go.Figure(fig_dict)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)   # 'axis equal'
    return fig

#plot_trajectory('experiments/experiment_0000/result_trajectory.csv')
