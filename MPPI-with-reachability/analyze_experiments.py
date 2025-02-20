from pathlib import Path
import pandas as pd

from experiment_storage import ExperimentStorage


def get_stats_for_configs(all_exp_df:pd.DataFrame) -> pd.DataFrame:

    # Each stat refers to a collection of trials with same # of samples & controller type
    grouped = all_exp_df.groupby(['mppi_samples', 'control_profile'])

    config_results = []

    for (num_samples, control_profile), group in grouped:

        num_trials = len(group)

        results = {
            'samples':          num_samples,
            'profile name':     control_profile,
            'num trials':       num_trials,
            'crashed (%)':      100 * group['crashed'     ].sum() / num_trials,
            'reached goal (%)': 100 * group['goal_reached'].sum() / num_trials,
            'absolute cost (avg ± std)':  f"{group['total_cost'   ].mean():.1f} ± {group['total_cost'   ].std():.1f}",
            'relative cost* (avg ± std)': f"{group['relative_cost'].mean():.3f} ± {group['relative_cost'].std():.3f}",
        }

        config_results.append(results)

    # Convert the list of results to a DataFrame for summary
    return pd.DataFrame(config_results)


def compute_relative_costs(
        df:pd.DataFrame,
        groupby_keys:list[str]=['mppi_samples', 'init_state'],
        reference_controller:str='Sample-safe MPPI (our method)'
    ) -> pd.DataFrame:
    """
    Returns dataframe with relative cost computation.
    This computation is performed relative to reference_controller
    when the dataframe is grouped by the keys in groupby_keys.
    """
    def update_with_relative_cost(group):
        # Check if all controllers succeeded on the episode
        if group['goal_reached'].all():
            # Get the cost of our method to use as reference for relative costs
            reference_cost = group.loc[(group['control_profile'] == reference_controller), 'total_cost'].item()
            group['relative_cost'] = group['total_cost'] / reference_cost
        else:
            # If not all controllers successful, return NaN for the entire group
            # NOTE: Pandas complains about this operation
            group['relative_cost'] = pd.NA
        return group

    # Pandas got upset if we passed columns with lists, so convert all lists to tuples
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(tuple)

    # Ensure that our reference controller is in the dataframe
    assert reference_controller in list(df['control_profile']), 'Reference controller not in data!'

    result_df = df.groupby(groupby_keys, group_keys=False).apply(update_with_relative_cost)

    # Revert grouping
    result_df.reset_index(drop=True, inplace=True)

    return result_df


def load_experiment_results(exp_dir:Path) -> pd.DataFrame:
    exps = [ExperimentStorage(x).get_all_experiment_info() for x in sorted(exp_dir.iterdir())]
    return pd.DataFrame(exps)


def create_experiment_batch_summary(experiments_dir:Path) -> pd.DataFrame:
    batch_summary_df = load_experiment_results(experiments_dir)
    batch_summary_df = compute_relative_costs(batch_summary_df)
    return batch_summary_df


def analyze_experiment_batch(experiments_dir) -> pd.DataFrame:
    exp_summaries_cache_fname = experiments_dir / '_summaries.csv'
    exp_stats_fname           = experiments_dir / '_stats.csv'

    try:
        all_exp_df = pd.read_csv( exp_summaries_cache_fname )
    except FileNotFoundError:
        all_exp_df = create_experiment_batch_summary(experiments_dir)
        all_exp_df.to_csv(exp_summaries_cache_fname, index=False)
    print(f"{len(all_exp_df)} experiments loaded")

    stats = get_stats_for_configs(all_exp_df)
    stats.to_csv(exp_stats_fname, index=False)
    print(stats)
    return stats


if __name__ == "__main__":

    experiments_dir = Path('experiments')
    analyze_experiment_batch(experiments_dir)
