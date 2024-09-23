from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os
from omegaconf import OmegaConf

import plot_traj
from experiment_storage import ExperimentStorage


# Function to get the list of available experiments from a directory
def get_experiment_list(directory):
    # Might want to filter for experiments that are finished
    return sorted([f for f in os.listdir(directory)])   # if f.isidr()

# Function to get the result data in a readable format
def format_dict_for_display(stored_result: ExperimentStorage):
    return f"RESULT:\n\n{OmegaConf.to_yaml(stored_result.get_summary())}\n\n" + \
           f"CONFIG:\n\n{OmegaConf.to_yaml(stored_result.get_config())}"



# Initialize Dash app
app = Dash(title='MPPI+reachability results')

# Layout
# These defaults are updated by callbacks
app.layout = html.Div([
    html.H2("Experiment Results Viewer"),

    # Dropdown to select experiment
    html.Label('Select experiment:'),
    dcc.Dropdown(id='experiment-dropdown', options=[]),

    # Slider to select timestep
    html.Label('Select timestep:'),
    dcc.Slider(id='timestep-slider', min=0, max=0, step=1, value=0, marks={}),

    # Plot
    html.Div([
        dcc.Graph(id='experiment-plot', style={'height': '80vh'}),
    ], style={'display': 'inline-block', 'width': '70%'}),

    # Results & config display
    html.Div([
        html.H4("Experiment Details"),
        html.Pre(id='details-display', style={'font-family': 'monospace'})
    ], style={'display': 'inline-block', 'width': '25%', 'vertical-align': 'top', 'padding-left': '20px'})

], style={'font-family': 'sans-serif'})




# Callback to update dropdown options based on available experiments
# Might not need this TBH, but nice that it auto-updates as experiments finish
@app.callback(
    Output('experiment-dropdown', 'options'),
    Input('experiment-dropdown', 'value')
)
def update_experiment_dropdown(value):
    experiment_dir = 'experiments'
    experiments = get_experiment_list(experiment_dir)
    return [{'label': exp, 'value': os.path.join(experiment_dir, exp)} for exp in experiments]


# Callback to update timestep slider based on selected experiment
@app.callback(
    Output('timestep-slider', 'max'),
    Output('timestep-slider', 'marks'),
    Input('experiment-dropdown', 'value')
)
def update_timestep_slider(selected_experiment):
    if selected_experiment is None:
        return 0, {}

    num_timesteps = ExperimentStorage(selected_experiment).get_num_timesteps()

    marks = {i: str(i) for i in range(num_timesteps)}
    return num_timesteps - 1, marks


# Callback to display the experiment result for the selected timestep
@app.callback(
    Output('experiment-plot', 'figure'),
    Output('details-display', 'children'),
    Input('experiment-dropdown', 'value'),
    Input('timestep-slider', 'value')
)
def display_timestep(selected_experiment, selected_timestep):
    if selected_experiment is None:
        return go.Figure(), 'No experiment selected.'

    stored_result = ExperimentStorage(selected_experiment)
    fig = plot_traj.plot_experiment_at_timestep(stored_result, selected_timestep)

    details_str = format_dict_for_display(stored_result)

    return fig, details_str
    #return dcc.Graph(figure=fig, style={'width': '98vw', 'height': '80vh'})

    # return f'Displaying raw data for timestep {selected_timestep}: {step_data}'


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
