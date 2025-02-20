from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from omegaconf import OmegaConf
from pathlib import Path
import sys   # argv

import plot_traj
from experiment_storage import ExperimentStorage


# Function to get the list of available experiments from a directory
def get_experiment_list(exp_directory:Path):
    # Might want to filter for experiments that are finished
    return sorted(exp_directory.iterdir())

def get_experiment_groups(exps):
    groups = [str(x)[:str(x).find('control')] for x in exps if 'control-0' in str(x)]
    return groups

# Function to get the result data in a readable format
def format_dict_for_display(stored_result: ExperimentStorage):
    return f"RESULT:\n\n{OmegaConf.to_yaml(stored_result.get_summary())}\n\n" + \
           f"CONFIG:\n\n{OmegaConf.to_yaml(stored_result.get_config())}"


# Initialize Dash app
app = Dash(title='DualGuard MPPI results')

# Layout
# These defaults are updated by callbacks
app.layout = html.Div([
    html.H2("Comparison across controllers"),

    # Dropdown to select experiment
    html.Label('Select experiment:'),
    dcc.Dropdown(id='experiment-dropdown', options=[]),

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
    groups = get_experiment_groups(get_experiment_list(experiments_path))
    return [{'label': str(g), 'value': str(g)} for g in groups]


# Callback to display the experiment result for the selected timestep
@app.callback(
    Output('experiment-plot', 'figure'),
    Output('details-display', 'children'),
    Input('experiment-dropdown', 'value'),
)
def display_compared_trajectories(selected_group):

    if selected_group is None:
        return go.Figure(), 'No experiment selected.'

    exps_in_group = [ExperimentStorage(x) for x in get_experiment_list(experiments_path) if selected_group in str(x)]
    fig = plot_traj.plot_experiment_comparison(exps_in_group)

    details_str = ''

    return fig, details_str


# Run the app
if __name__ == '__main__':
    if len(sys.argv) == 2:
        experiments_path = Path(sys.argv[1])
    else:
        print("*** Attempting to default to '/experiments' as experiment path.       ***")
        print("*** Otherwise, specify the experiment batch directory as an argument. ***")
        experiments_path = Path('experiments')

    app.run_server(debug=True)
