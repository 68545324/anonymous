import copy

from ray import tune

from submission.utils import (
    log,
    miscellaneous,
    aggregate_and_plot_tensorboard_data,
)


def get_hyperparameters(debug, train_n_replicates=None, env=None):

    if train_n_replicates is None:
        train_n_replicates = 2 if debug else int(3 * 2)
    seeds = miscellaneous.get_random_seeds(train_n_replicates)

    exp_name, _ = log.log_in_current_day_dir("SOS")

    hparams = {
        "debug": debug,
        "load_plot_data": None,
        "exp_name": exp_name,
        "classify_into_welfare_fn": True,
        "train_n_replicates": train_n_replicates,
        "env_name": "IteratedAsymBoS" if env is None else env,
        "lr": 1.0 / 10,
        "gamma": 0.96,
        "num_epochs": 5 if debug else 100,
        # "method": "lola",
        "method": "sos",
        "inital_weights_std": 1.0,
        "seed": tune.grid_search(seeds),
        "metric": "mean_reward_player_row",
        "plot_keys": aggregate_and_plot_tensorboard_data.PLOT_KEYS
        + ["mean_reward"],
        "plot_assemblage_tags": aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
        + [("mean_reward",)],
        "x_limits": (-0.1, 4.1),
        "y_limits": (-0.1, 4.1),
        "max_steps_in_eval": 100,
    }

    return hparams


def get_tune_config(hp: dict):
    tune_config = copy.deepcopy(hp)
    assert tune_config["env_name"] in ("IPD", "IteratedAsymBoS")
    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": hp["max_steps_in_eval"],
        "get_additional_info": True,
    }
    tune_config["plot_axis_scale_multipliers"] = (
        (
            1 / hp["max_steps_in_eval"],
            1 / hp["max_steps_in_eval"],
        ),
    )
    if "num_episodes" in tune_config:
        stop_config = {"episodes_total": tune_config["num_episodes"]}
    else:
        stop_config = {"episodes_total": tune_config["num_epochs"]}

    return tune_config, stop_config, env_config
