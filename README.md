# DualGuard-MPPI

This is the authors' implementation of [DualGuard MPPI (arXiv preprint)](https://arxiv.org/abs/2502.01924) by Javier Borquez, Luke Raus, Yusuf Umut Ciftci, and Somil Bansal.

Specifically, this repository can be used to replicate the Safe Planar Navigation experiments (section V-A of the paper) with minimal setup.

## Setup

We recommend first creating a python virtual environment.

Then, intall external dependencies via:

```
pip install -r requirements.txt
```

## Running experiments

From this repository's main directory:

```
cd MPPI-with-reachability
python run_experiments.py
```

To test different controller configurations, you can modify the options in `run_many_experiments.py`.

All experiments are specified with a `config.yaml` file that lists all the relevant (hyper)parameters for that experiment. Once each experiment is run, it produces a `result_summary.yaml` file which summarizes the result, and stores more detailed data in `result_details.hdf5` (both of which will be placed in the experiment's directory).

## Visualizing results

The results can be visualized via `plot_experiment_results.py` and using the Dash GUI to select and experiment & control timestep.

## Analyzing collective results

Statistics across many trials can be produced using `analyze_experiments.py`.

Cost distribution plots can be produced using `plot_cost_distributions.py`.

## (TODO: Update this for .h5) Interface descriptions

The experiment themselves rely on a single data file: the BRT file. This is generated by [`matlab/export_3d_brt_with_gradient.m`](matlab/export_3d_brt_with_gradient.m), which is hopefully fairly self-explanatory. It contains:

- the BRT value function
- the BRT value gradient with respect to theta (used to compute the optimal control)
- the grid axes over which the BRT value was calculated
- the BRT 'initial value,' i.e. the SDF of the obstacles (commonly `data0` in helperOC; this is used to compute whether the system has crashed into an obstacle)

The MATLAB routine which creates and saves this file does a few manipulations specific to this system to save some effort on the Python side; for example, I pad the borders of the X & Y dimensions of the state space with obstacle, and I duplicate the theta=0 slice of the value function to the end at theta=pi to handle theta's periodicity.

Additionally, my set of experiments relies on some state pairs stored in [a JSON file](config_data/dubin_environment_state_pairs.json). This is simply a dictionary where `init` and `goal` are keys which each point to a list of state vectors.
