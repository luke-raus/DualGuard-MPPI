from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd

from experiment_result import ExperimentResult


experiments_path = Path('experiments')
control_profiles_fname = Path('config').joinpath('control_profiles.yaml')



control_profiles = OmegaConf.load(control_profiles_fname)

experiments_info = [ExperimentResult(x).get_all_experiment_info() for x in sorted(experiments_path.iterdir())]
exp_df = pd.DataFrame(experiments_info)
print(exp_df.keys())

num_samples_settings = exp_df['mppi_samples'].unique().tolist()

config_results = []
for num_samples in num_samples_settings:
    for profile_name, settings in control_profiles.items():

        config_trials = exp_df.loc[ (exp_df['control_profile'] == profile_name) ]
        # Alternatively, filter for all relevant settings using filter_df_by_dict()

        num_trials = len(config_trials)

        results = {
            'profile name': profile_name,
            'num trials':  num_trials,
            'cost (avg ± std)': f"{config_trials['total_cost'].mean():.1f} ± {config_trials['total_cost'].std():.1f}",
            'crashed (%)':      100 * config_trials['crashed'].sum() / num_trials,
            'reached goal (%)': 100 * config_trials['goal_reached'].sum() / num_trials,
        }
        config_results.append(results)

    summary = pd.DataFrame(config_results)

    print(f'\nSamples: {num_samples}')
    print(summary)


def filter_df_by_dict(df: pd.DataFrame, filter_dict: dict) -> pd.DataFrame:
    # See: https://stackoverflow.com/a/34162576
    return df.loc[(df[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)]
