# MPPI-with-reachability

## Run an experiment

Experiments should be able to be run via `run_experiments.py`.

This script loads all the relevant data, sets up the dynamics/map/controller objects, and then loops over (potentially) different trials and controller configurations, calling `run_trial.py` which actually executes the simulated experiment.

It then saves an experiment data object as a `.pkl` file to the `/results` directory.

## Plot an experiment

Experiment results can be plotted via `plot_example.py`, which loads the data for an experiment and then calls the plotting routine in `plot_traj_with_brt.py`. Most importantly, you can set which experiment file to plot!
