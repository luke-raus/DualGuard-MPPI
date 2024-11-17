from flask.cli import F
import plotly.graph_objects as go
import numpy as np
import h5py
from pathlib import Path

from experiment_storage import ExperimentStorage


#class ExperimentVisualization:

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

    new_shapes = tuple(obs_rects) + tuple(obs_circs)

    if "shapes" in layout.keys():
        layout["shapes"] += new_shapes
    else:
        layout["shapes"] = new_shapes

    return layout


def update_layout_with_goal_state(layout:dict, config) -> dict:
    x = config.goal_state[0]
    y = config.goal_state[1]
    r = config.goal_state_threshold
    new_shapes = ({'x0': float(x-r), 'y0': float(y-r), 'x1': float(x+r), 'y1': float(y+r),
                   'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'blue', 'opacity': 0.5, 'line': {'width': 0}},)

    if "shapes" in layout.keys():
        layout["shapes"] += new_shapes
    else:
        layout["shapes"] = new_shapes

    return layout


def get_trace_of_overall_trajectory_to_index(result:ExperimentStorage, index:int = -1, color="black", markers=True, name="System trajectory") -> dict:

    trajectory = result.get_overall_trajectory()
    if index == -1:
        index = np.shape(trajectory)[0]

    trajectory_trace = {
        "x": trajectory[:index+1, 0],
        "y": trajectory[:index+1, 1],
        "mode": "lines+markers",
        "line": {
            "color": color
        },
        "name": name,
        "showlegend": True,
    }
    if markers:
        trajectory_trace["marker"] = {
            "symbol": "triangle-up-open",
            "size": 15,
            "angle": convert_angles_for_plot(trajectory[:index+1, 2])
        }
    return trajectory_trace


def convert_angles_for_plot(state_angles):
     return -(state_angles * 180 / 3.14159264) + 90


def get_traces_of_samples(step_data:dict, max_samples=100) -> list[dict]:

    sample_states  = step_data['sample_states']   # (K, T, nx)
    sample_costs   = step_data['sample_costs']    # (K)
    sample_weights = step_data['sample_weights']  # (K)

    num_samples = min(np.shape(sample_states)[0], max_samples)

    sample_alpha = 0.4

    # We group the samples into those that are basically "discarded" or not.
    # Those that are discarded, color orange.
    # Those that are not, a gradient: higher weight = blue, lower weight = red.

    discarded_weight_threshold = 1e-9   # We assume that any sample with less than this much weight is effectively "discarded"

    is_discarded = sample_weights < discarded_weight_threshold
    print(is_discarded)
    print(all(is_discarded))

    if not all(is_discarded):
        max_weight = max(sample_weights)
        min_weight_above_threshold = min(sample_weights[ ~is_discarded ])

        print(max_weight)
        print(min_weight_above_threshold)

        sample_weights_scaled = (sample_weights - min_weight_above_threshold) / (max_weight - min_weight_above_threshold + 1e-9)  # avoid divide by zero if min=max

    sample_colors = [ f"rgba(255,140,16,{sample_alpha})" if is_discarded[i]
                      else f"rgba({255-round(255*sample_weights_scaled[i])},0,{round(255*sample_weights_scaled[i])},{sample_alpha})"
                      for i in range(len(sample_weights))]

    print(sample_states[0][:, 0])
    print(sample_states[0][:, 1])

    sample_traces = [ {
        "x": sample_states[i][:, 0],
        "y": sample_states[i][:, 1],
        "mode": "lines+markers",
        "showlegend": False,
        "line": {"color": sample_colors[i]},
        "name": f"sample {i}",
        "text": f"cost: {round(sample_costs[i], 1)}, weight: {sample_weights[i]}",
    } for i in range(num_samples) ]

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
def plot_experiment_at_timestep(result:ExperimentStorage, step_index:int) -> go.Figure:

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
    layout = update_layout_with_goal_state( layout, result.get_config() )

    layout["xaxis"] = {'showgrid': False, 'zeroline': False}
    layout["yaxis"] = {'showgrid': False, 'zeroline': False}

    fig_dict = { "data": traces, "layout": layout }
    fig = go.Figure(fig_dict)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)   # 'axis equal'
    return fig


def plot_experiment_comparison(results:list[ExperimentStorage]) -> go.Figure:

    name_map = {
        'Vanilla MPPI with obstacle costs':  ('Obstacle penalty',    '#a4cde1'),   # light blue
        'Vanilla MPPI with BRT costs':       ('BRT penalty',         '#1e78b0'),   # blue
        'Filtered MPPI with obstacle costs': ('Obstacle pen. + LRF', '#b1dd8e'),   # light green
        'Filtered MPPI with BRT costs':      ('BRT pen. + LRF',      '#369d3b'),   # green
        'Shield MPPI':                       ('Shield-MPPI',         '#694296'),   # purple
        'Sample-safe MPPI (our method)':     ('Our method',          '#ff7e1e')    # orange
    }

    traj_names = [res.get_config()['control_profile'] for res in results]
    fin_and_costs = [(res.get_summary()['goal_reached'], res.get_summary()['total_cost']) for res in results]

    legend_entries = [f'{name_map[x][0]}, {fin_and_costs[i]}' for i, x in enumerate(traj_names)]
    colors = [name_map[x][1] for x in traj_names]

    traces = [get_trace_of_overall_trajectory_to_index(res, name=legend_entries[i], color=colors[i], markers=False)
              for i, res in enumerate(results)]

    # Assumes all results share same map
    layout = update_plot_layout_with_map( {}, results[0].get_environment_path() )
    layout = update_layout_with_goal_state( layout, results[0].get_config() )

    layout["xaxis"] = {'showgrid': False, 'zeroline': False}
    layout["yaxis"] = {'showgrid': False, 'zeroline': False}

    fig_dict = { "data": traces, "layout": layout }
    fig = go.Figure(fig_dict)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)   # 'axis equal'
    return fig
