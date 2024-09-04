from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd

def get_expr_info(experiment_dir:Path) -> dict:
    config = OmegaConf.load(experiment_dir / 'config.yaml')
    result = OmegaConf.load(experiment_dir / 'result_summary.yaml')
    # Merge dictionaries
    return {'path':str(experiment_dir), **config, **result}


if __name__ == "__main__":

    experiments_path = Path('experiments')
    experiment_info = [get_expr_info(x) for x in sorted(experiments_path.iterdir())]
    exp_df = pd.DataFrame(experiment_info)
    print(exp_df)
    print()

    configs = [
        ('MPPI with obs costs',        False,  'obs'),
        ('MPPI with BRT costs',        False,  'brt'),
        ('Our method with obs costs',  True,   'obs'),
        ('Our method with BRT costs',  True,   'brt')
    ]

    for config in configs:
        name, filter_samples, cost_type = config

        sub_df = exp_df.loc[ (exp_df['apply_safety_filter_to_samples'] == filter_samples) &
                             (exp_df['cost_from_obstacles_or_BRT'] == cost_type) ]

        num_trials = len(sub_df)

        avg_cost = sub_df['total_cost'].mean()
        std_cost = sub_df['total_cost'].std()

        num_trials_crashed      = sub_df['crashed'].sum()
        num_trials_reached_goal = sub_df['goal_reached'].sum()

        percent_trials_crashed      = 100 * num_trials_crashed      / num_trials
        percent_trials_reached_goal = 100 * num_trials_reached_goal / num_trials

        print(f'Config: {name}   -   {num_trials} trials   -    % crashed: {percent_trials_crashed}   -   % reached goal: {percent_trials_reached_goal}    -   avg cost: {avg_cost:.1f} +/- {std_cost:.1f}')
