import copy
import os
import argparse

import ray
from ray import tune

from ray.rllib.agents.pg import PGTorchPolicy

from submission.algos.lola.train_exact_tune_class_API import LOLAExactTrainer
from submission.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
    IteratedAsymBoS,
    IteratedAsymBoSandPD,
)
from submission.base_game_experiments import lola_pg_official
from submission.utils import (
    policy,
    log,
    miscellaneous,
    aggregate_and_plot_tensorboard_data,
)

EGALITARIAN = "egalitarian"
MIXED = "mixed"
UTILITARIAN = "utilitarian"
FAILURE = "failure"


def main(debug):
    hparams = get_hyperparameters(debug)

    if hparams["load_plot_data"] is None:
        ray.init(
            num_cpus=1,
            num_gpus=0,
            local_mode=debug,
        )
        experiment_analysis_per_welfare = train(hparams)
    else:
        experiment_analysis_per_welfare = None
    ray.shutdown()
    ray.init(
        num_cpus=os.cpu_count(),
        num_gpus=0,
        local_mode=debug,
    )
    evaluate(experiment_analysis_per_welfare, hparams)
    ray.shutdown()


def get_hyperparameters(debug, train_n_replicates=None, env=None):
    """Get hyperparameters for LOLA-Exact for matrix games"""

    if not debug:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--env",
            type=str,
            choices=[
                "IteratedPrisonersDilemma",
                "IteratedAsymBoS",
                "IteratedAsymBoSandPD",
            ],
            help="Env to use.",
        )
        parser.add_argument("--train_n_replicates", type=int)
        args = parser.parse_args()
        args = args.__dict__
        if "env" in args.keys():
            env = args["env"]
        if "train_n_replicates" in args.keys():
            train_n_replicates = args["train_n_replicates"]

    if train_n_replicates is None:
        train_n_replicates = 2 if debug else int(3 * 10)
    seeds = miscellaneous.get_random_seeds(train_n_replicates)

    exp_name, _ = log.log_in_current_day_dir("LOLA_Exact")

    hparams = {
        "debug": debug,
        "load_plot_data": None,
        # Example "load_plot_data": ".../SelfAndCrossPlay_save.p",
        "exp_name": exp_name,
        "classify_into_welfare_fn": True,
        "train_n_replicates": train_n_replicates,
        # "env_name": "IteratedPrisonersDilemma" if env is None else env,
        # "env_name": "IteratedAsymBoS" if env is None else env,
        "env_name": "IteratedAsymBoSandPD" if env is None else env,
        "num_episodes": 1 if debug else 1,
        "trace_length": 200 if debug else 200,
        "re_init_every_n_epi": 1,
        "simple_net": True,
        "corrections": True,
        "pseudo": False,
        "num_hidden": 32,
        "reg": 0.0,
        "lr": 1.0,
        "lr_correction": 1.0,
        "gamma": 0.96,
        "seed": tune.grid_search(seeds),
        "metric": "ret1",
        "with_linear_LR_decay_to_zero": False,
        "clip_update": None,
        # "with_linear_LR_decay_to_zero": True,
        # "clip_update": 0.1,
        # "lr": 0.001,
        "plot_keys": aggregate_and_plot_tensorboard_data.PLOT_KEYS + ["ret"],
        "plot_assemblage_tags": aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
        + [("ret",)],
        "x_limits": (-0.1, 4.1),
        "y_limits": (-0.1, 4.1),
    }

    hparams["plot_axis_scale_multipliers"] = (
        1 / hparams["trace_length"],
        1 / hparams["trace_length"],
    )
    return hparams


def train(hp):
    tune_config, stop_config, _ = get_tune_config(hp)
    # Train with the Tune Class API (not an RLLib Trainer)
    experiment_analysis = tune.run(
        LOLAExactTrainer,
        name=hp["exp_name"],
        config=tune_config,
        checkpoint_at_end=True,
        stop=stop_config,
        metric=hp["metric"],
        mode="max",
    )
    if hp["classify_into_welfare_fn"]:
        experiment_analysis_per_welfare = (
            _classify_trials_in_function_of_welfare(experiment_analysis, hp)
        )
    else:
        experiment_analysis_per_welfare = {"": experiment_analysis}

    return experiment_analysis_per_welfare


def get_tune_config(hp: dict):
    tune_config = copy.deepcopy(hp)
    assert tune_config["env_name"] in (
        "IteratedPrisonersDilemma",
        "IteratedAsymBoS",
        "IteratedAsymBoSandPD",
    )

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": tune_config["trace_length"],
        "get_additional_info": True,
    }

    if tune_config["env_name"] in ["IteratedAsymBoS", "IteratedAsymBoSandPD"]:
        tune_config["Q_net_std"] = 3.0
    else:
        tune_config["Q_net_std"] = 1.0

    if tune_config["env_name"] in (
        "IteratedPrisonersDilemma",
        "IteratedAsymBoS",
        "IteratedAsymBoSandPD",
    ):
        tune_config["gamma"] = (
            0.96 if tune_config["gamma"] is None else tune_config["gamma"]
        )
        tune_config["save_dir"] = "dice_results_ipd"

    stop_config = {"episodes_total": tune_config["num_episodes"]}
    return tune_config, stop_config, env_config


def evaluate(experiment_analysis_per_welfare, hp):
    (
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    ) = generate_eval_config(hp)

    lola_pg_official._evaluate_self_and_cross_perf(
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
        experiment_analysis_per_welfare,
        n_cross_play_per_checkpoint=min(15, hp["train_n_replicates"] - 1)
        if hp["classify_into_welfare_fn"]
        else None,
    )


def generate_eval_config(hp):
    hp_eval = copy.deepcopy(hp)

    hp_eval["min_iter_time_s"] = 3.0
    hp_eval["seed"] = miscellaneous.get_random_seeds(1)[0]
    hp_eval["batch_size"] = 1
    hp_eval["num_episodes"] = 100

    tune_config, stop_config, env_config = get_tune_config(hp_eval)
    tune_config["TuneTrainerClass"] = LOLAExactTrainer

    hp_eval["group_names"] = ["lola"]
    hp_eval["scale_multipliers"] = (
        1 / tune_config["trace_length"],
        1 / tune_config["trace_length"],
    )
    hp_eval["jitter"] = 0.05

    if hp_eval["env_name"] == "IteratedPrisonersDilemma":
        hp_eval["env_class"] = IteratedPrisonersDilemma
        hp_eval["x_limits"] = (-3.5, 0.5)
        hp_eval["y_limits"] = (-3.5, 0.5)
    elif hp_eval["env_name"] == "IteratedAsymBoS":
        hp_eval["env_class"] = IteratedAsymBoS
        hp_eval["x_limits"] = (-0.1, 4.1)
        hp_eval["y_limits"] = (-0.1, 4.1)
    elif hp_eval["env_name"] == "IteratedAsymBoSandPD":
        hp_eval["env_class"] = IteratedAsymBoSandPD
        # hp_eval["x_limits"] = (-3.1, 4.1)
        # hp_eval["y_limits"] = (-3.1, 4.1)
        hp_eval["x_limits"] = (-6.1, 5.1)
        hp_eval["y_limits"] = (-6.1, 5.1)
    else:
        raise NotImplementedError()

    rllib_config_eval = {
        "env": hp_eval["env_class"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    policy.get_tune_policy_class(PGTorchPolicy),
                    hp_eval["env_class"](env_config).OBSERVATION_SPACE,
                    hp_eval["env_class"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)},
                ),
                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(PGTorchPolicy),
                    hp_eval["env_class"](env_config).OBSERVATION_SPACE,
                    hp_eval["env_class"].ACTION_SPACE,
                    {"tune_config": copy.deepcopy(tune_config)},
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": hp_eval["seed"],
        "min_iter_time_s": hp_eval["min_iter_time_s"],
        "num_workers": 0,
        "num_envs_per_worker": 1,
    }

    policies_to_load = copy.deepcopy(env_config["players_ids"])
    trainable_class = LOLAExactTrainer

    return (
        hp_eval,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    )


def _classify_trials_in_function_of_welfare(experiment_analysis, hp):
    experiment_analysis_per_welfare = {}
    for trial in experiment_analysis.trials:
        welfare_name = _get_trial_welfare(trial, hp)
        if welfare_name == FAILURE:
            continue
        if welfare_name not in experiment_analysis_per_welfare.keys():
            _add_empty_experiment_analysis(
                experiment_analysis_per_welfare,
                welfare_name,
                experiment_analysis,
            )
        experiment_analysis_per_welfare[welfare_name].trials.append(trial)
    return experiment_analysis_per_welfare


def _get_trial_welfare(trial, hp):
    reward_player_1 = trial.last_result["ret1"]
    reward_player_2 = trial.last_result["ret2"]
    welfare_name = classify_into_welfare_based_on_rewards(
        reward_player_1, reward_player_2, hp
    )
    return welfare_name


def classify_into_welfare_based_on_rewards(
    reward_player_1, reward_player_2, hp
):

    if hp["env_name"] == "IteratedAsymBoSandPD":
        if reward_player_1 > 3.5 and reward_player_2 < 1.5:
            return UTILITARIAN
        elif reward_player_2 < 2.0 and reward_player_1 > 2.5:
            return MIXED
        elif reward_player_2 > 1.5 and reward_player_1 > 1.5:
            return EGALITARIAN
        else:
            return FAILURE
    else:
        ratio = reward_player_1 / reward_player_2
        if ratio < 1.5:
            return EGALITARIAN
        elif ratio < 2.5:
            return MIXED
        else:
            return UTILITARIAN


def _add_empty_experiment_analysis(
    experiment_analysis_per_welfare, welfare_name, experiment_analysis
):
    experiment_analysis_per_welfare[welfare_name] = copy.deepcopy(
        experiment_analysis
    )
    experiment_analysis_per_welfare[welfare_name].trials = []


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
