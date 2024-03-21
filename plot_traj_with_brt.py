import plotly.graph_objects as go
import numpy as np


def plotly_start_goal_location(start_state, goal_state):
    """
    start_state is 1d array-like
    goal_state is 1d array-like
    """
    # use via .extend()

    # start & goal positition
    return [{"x": [start_state[0]],
             "y": [start_state[1]],
             "mode": "markers",
             "showlegend": False,
             "marker": dict(size=20, color='blue'),
             "name": "START"},
            {"x": [goal_state[0]],
             "y": [goal_state[1]],
             "mode": "markers",
             "showlegend": False,
             "marker": dict(size=20, color='purple'),
             "name": "GOAL"}
            ]


def plotly_samples(sampled_states, sample_costs, sample_weights, max_samples=100):
    """
    frame["data"].extend( THIS )
    """

    # frame = {"data": [], "name": str(t)}
    frame_data = []

    # sampled_states  = expr_data['sampled_states'][t]            # (K, T, nx)
    # sample_costs    = expr_data['sample_costs'][t]              # (K,)
    # sample_weights  = expr_data['sample_weights'][t].tolist()   # (K,)

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

    # max_samples = 30
    nSamples = min(np.shape(sampled_states)[0], max_samples)

    for sample_index in range(nSamples):

        sample_cost = sample_costs[sample_index]
        sample_weight = sample_weights[sample_index]

        sample_alpha = 0.4

        # If hit an obstacle, make orange
        if sample_cost > cost_threshold:
            sample_color = f"rgba(255,140,16,{sample_alpha})"
        else:
            # maps to 0-1, then to 0-255
            sample_cost_scaled = round(
                255*float((sample_cost - min_cost) / range_cost))
            sample_color = f"rgba({sample_cost_scaled},0,{255-sample_cost_scaled},{sample_alpha})"

        data_dict = {
            "x": sampled_states[sample_index][:, 0].tolist(),
            "y": sampled_states[sample_index][:, 1].tolist(),
            "mode": "lines",
            "showlegend": False,
            "line": dict(color=sample_color),
            "name": str(sample_index),
            "text": "cost: " + str(round(sample_cost, 1)) + ", weight: " + str(round(sample_weight, 6)),
        }
        # if t == 0:  # figure wants initial set of data in "data" attribute
        #     fig_dict["data"].append(data_dict)
        frame_data.append(data_dict)

    return frame_data


def plot_trajectory_with_brt_NEW(map_data, expr_data, brt_dict, start_goal_data, show_brt=True, title=None, trial=0, frames_to_plot=None, max_samples=200, output_height=1000):

    # trial = 0

    # maxSamples = 100
    maxTimesteps = 400

    nTimesteps = min(len(expr_data['nom_states']), maxTimesteps)

    # Boolean of whether or not sampled states exist in data
    samples = 'sampled_states' in expr_data.keys()
    if samples:
        nSamples_saved = np.shape(expr_data['sampled_states'][0])[0]
        nSamples = min(nSamples_saved, max_samples)

    # BRT metadata for grabbing closest slice
    # n_brt_thetas = brt_tensor.shape[-1]
    # brt_thetas = torch.linspace(start=-torch.pi, end=torch.pi, steps=n_brt_thetas)
    brt_x = brt_dict['grid_axes'][0]
    brt_y = brt_dict['grid_axes'][1]
    brt_thetas = np.array(brt_dict['grid_axes'][2])

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"]["xaxis"] = {'showgrid': False, 'zeroline': False}
    fig_dict["layout"]["yaxis"] = {'showgrid': False, 'zeroline': False}

    fig_dict["layout"]["updatemenus"] = [
        {"buttons": [
            {"args": [None, {"frame": {"duration": 5, "redraw": False},
                             "fromcurrent": True, "transition": {"duration": 0}}],
             "label": "Play",
             "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate",
                               "transition": {"duration": 0}}],
             "label": "Pause",
             "method": "animate"}],
         "direction": "left", "pad": {"r": 10, "t": 87}, "showactive": False, "type": "buttons",
         "x": 0.1, "xanchor": "right", "y": 0, "yanchor": "top"
         }]

    sliders_dict = {
        "active": 0, "yanchor": "top", "xanchor": "left",
        "currentvalue": {"prefix": "Timestep: ", "visible": True, "xanchor": "left"},
        "pad": {"b": 10, "t": 50}, "len": 0.9, "x": 0.1, "y": 0, "steps": []
    }

    if frames_to_plot is None:
        frames_to_plot = range(1, nTimesteps)

    for t in frames_to_plot:

        frame = {"data": [], "name": str(t)}
        fd = []

        fd.extend(plotly_start_goal_location(start_state=start_goal_data['init'][trial],
                                             goal_state=start_goal_data['goal'][trial]))

        nom_traj = expr_data['nom_states'][t]

        if samples:
            fd.extend(plotly_samples(sampled_states=expr_data['sampled_states'][t],
                                     sample_costs=expr_data['sample_costs'][t],
                                     sample_weights=expr_data['sample_weights'][t].tolist(
            ),
                max_samples=max_samples))

        # Plot nominal trajectory at timestep
        fd.append(
            {
                "x": nom_traj[:, 0].tolist(),
                "y": nom_traj[:, 1].tolist(),
                "mode": "lines",
                "showlegend": False,
                "line": dict(color='rgb(0,255,0)', width=5),
                "name": "Nominal trajectory"
            })

        # Plot actual trajectory taken
        fd.append({
            # [:t+1] means only plot currently taken
            "x": [st[0] for st in expr_data['actual_state'][:t+1]],
            "y": [st[1] for st in expr_data['actual_state'][:t+1]],
            "mode": "lines+markers",
            "showlegend": False,
            "line": dict(color='black'),
            "name": "Actual trajectory"
        })

        # Plot BRT slide at timestep
        if show_brt:
            curr_theta = expr_data['actual_state'][t][2]

            theta_ind = np.sum(brt_thetas < curr_theta)
            if np.abs(curr_theta - brt_thetas[theta_ind-1]) < np.abs(curr_theta - brt_thetas[theta_ind]):
                theta_ind -= 1

            fd.append(go.Heatmap(
                x=np.linspace(-5.0, 5.0, len(brt_x.tolist())).tolist(),
                y=np.linspace(-5.0, 5.0, len(brt_y.tolist())).tolist(),
                z=np.transpose(np.tensor(
                    brt_dict['value'][:, :, theta_ind]), 0, 1),
                colorscale='Blackbody',  # 'Aggrnyl',
                name='BRT',
                colorbar={'title': {'text': 'BRT value'}}
            ))

        frame["data"] = fd

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [t],
            {"frame": {"duration": 0, "redraw": True},
             "mode": "immediate",
             "transition": {"duration": 0}}
        ],
            "label": t,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    # Set figure data to data of first frame
    fig_dict["data"] = fig_dict["frames"][0]["data"]
    fig_dict["layout"]["sliders"] = [sliders_dict]

    # Get obstacle data from map
    obs_xyr = [(map_data['x'][i], map_data['y'][i], map_data['r'][i])
               for i in range(len(map_data['x']))]

    # Plot obstacles and walls as layout objects (permanent across frames)

    # kwargs = {'type':'circle', 'xref':'x', 'yref':'y', 'fillcolor':'gray', 'layer':'below', 'opacity':1.0}
    # points = [go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs) for x, y, r in obs_xyr]

    w = 5.    # wall distances from center
    w_t = 1.    # wall display thickness
    walls = [[-(w+w_t), w, -w, -w], [-(w+w_t), (w+w_t), (w+w_t), w],
             [-(w+w_t), -w, (w+w_t), -(w+w_t)], [w, w, (w+w_t), -w]]

    obs_circs = [{'x0': float(x-r), 'y0': float(y-r), 'x1': float(x+r), 'y1': float(y+r), 'type': 'circle', 'xref': 'x', 'yref': 'y',
                  'fillcolor': 'black', 'opacity': 0.5, 'line': {'width': 0}} for x, y, r in obs_xyr]

    obs_rects = [{'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'type': 'rect', 'xref': 'x', 'yref': 'y',
                  'fillcolor': 'black', 'opacity': 0.5, 'line': {'width': 0}} for x0, y0, x1, y1 in walls]

    obs_shapes = obs_circs + obs_rects
    fig_dict["layout"]["shapes"] = tuple(obs_shapes)

    if title is None:
        title = ""
    fig_dict["layout"]["title"] = title

    fig = go.Figure(fig_dict)

    # Akin to 'axis equal' in MATLAB
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(height=output_height)

    fig.show()
