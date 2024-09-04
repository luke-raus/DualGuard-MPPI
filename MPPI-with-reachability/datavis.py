import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import os
from experiment_result import ExperimentResult


# Initialize Dash app
app = dash.Dash(title='MPPI+reachability results')

# Function to get the list of available experiments from a directory
def get_experiment_list(directory):
    # Might want to filter for experiments that are finished
    return sorted([f for f in os.listdir(directory)])   # if f.isidr()

# Layout
# These defaults are updated by callbacks
app.layout = html.Div([
    html.H2("Experiment Results Viewer"),

    # Dropdown to select experiment
    html.Label('Select Experiment:'),
    dcc.Dropdown(id='experiment-dropdown', options=[]),

    # Slider to select timestep
    html.Label('Select Timestep:'),
    dcc.Slider(id='timestep-slider', min=0, max=0, step=1, value=0, marks={}),

    # Placeholder for displaying experiment results
    html.Div(id='experiment-display')
])


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

    num_timesteps = ExperimentResult(selected_experiment).get_num_timesteps()
    marks = {i: str(i) for i in range(num_timesteps)}
    return num_timesteps - 1, marks


# Callback to display the experiment result for the selected timestep
@app.callback(
    Output('experiment-display', 'children'),
    Input('experiment-dropdown', 'value'),
    Input('timestep-slider', 'value')
)
def display_experiment(selected_experiment, selected_timestep):
    if selected_experiment is None:
        return "No experiment selected."

    step_data = ExperimentResult(selected_experiment).load_timestep(selected_timestep)

    # Display result (customize based on what you're visualizing)
    return f"Displaying raw data for timestep {selected_timestep}: {step_data}"


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
