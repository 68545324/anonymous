import json
import logging
import os

import numpy as np
from ray.rllib.agents.pg import PGTrainer

from submission import utils
from submission.base_game_experiments import lola_exact_official
from submission.base_game_experiments.lola_exact_official import (
    UTILITARIAN,
    EGALITARIAN,
    MIXED,
)
from submission.meta_game_experiments.utils import helpers
from submission.utils import (
    restore,
    path,
)

logger = logging.getLogger(__name__)


def get_hyperparameters(debug):
    """Get hyperparameters for meta game with LOLA-Exact policies in base
    game"""
    # env = "IPD"
    env = "IteratedAsymBoS"

    hp = lola_exact_official.get_hyperparameters(
        debug, train_n_replicates=1, env=env
    )

    hp.update(
        {
            "n_replicates_over_full_exp": 2 if debug else 20,
            "final_base_game_eval_over_n_epi": 1 if debug else 200,
            "tau_range": np.arange(0.0, 1.1, 0.5)
            if hp["debug"]
            else np.arange(0.0, 1.1, 0.1),
            "n_self_play_in_final_meta_game": 0,
            "n_cross_play_in_final_meta_game": 1 if debug else 10,
            "welfare_functions": [
                (EGALITARIAN, EGALITARIAN),
                (MIXED, MIXED),
                (UTILITARIAN, UTILITARIAN),
            ],
        }
    )
    return hp


def _load_base_game_results(hp, load_base_replicate_i):
    # Base policies provided
    prefix = os.path.join(
        os.path.dirname(__file__),
        "../../base_game_experiments/results/LOLA_Exact",
    )
    # New base policies
    # prefix = "~/ray_results/LOLA_Exact"

    prefix = os.path.expanduser(prefix)
    if "IteratedAsymBoS" in hp["env_name"]:
        hp["data_dir"] = (
            os.path.join(prefix, "2021_05_07/07_52_32"),
            os.path.join(prefix, "2021_05_07/08_02_38"),
            os.path.join(prefix, "2021_05_07/08_02_49"),
            os.path.join(prefix, "2021_05_07/08_03_03"),
            os.path.join(prefix, "2021_05_07/08_54_58"),
            os.path.join(prefix, "2021_05_07/08_55_34"),
            os.path.join(prefix, "2021_05_07/09_04_07"),
            os.path.join(prefix, "2021_05_07/09_09_30"),
            os.path.join(prefix, "2021_05_07/09_09_42"),
            os.path.join(prefix, "2021_05_07/10_02_15"),
            os.path.join(prefix, "2021_05_07/10_02_30"),
            os.path.join(prefix, "2021_05_07/10_02_39"),
            os.path.join(prefix, "2021_05_07/10_02_50"),
            os.path.join(prefix, "2021_05_05/14_49_18"),
            os.path.join(prefix, "2021_05_05/14_50_39"),
            os.path.join(prefix, "2021_05_05/14_51_01"),
            os.path.join(prefix, "2021_05_05/14_53_56"),
            os.path.join(prefix, "2021_05_05/14_56_32"),
            os.path.join(prefix, "2021_05_05/15_46_08"),
            os.path.join(prefix, "2021_05_05/15_46_23"),
            os.path.join(prefix, "2021_05_05/15_46_59"),
            os.path.join(prefix, "2021_05_05/15_47_22"),
            os.path.join(prefix, "2021_05_05/15_48_22"),
        )[load_base_replicate_i]
    else:
        raise ValueError(f'bad env_name: {hp["env_name"]}')

    assert os.path.exists(hp["data_dir"]), (
        "Path doesn't exist. Probably that the prefix need to "
        f"be changed to fit the current machine used. path: {hp['data_dir']}"
    )

    print("==== Going to process data_dir", hp["data_dir"], "====")

    hp["ckpt_per_welfare"] = _get_checkpoints_for_each_welfare_in_dir(
        hp["data_dir"], hp
    )

    return hp


def _get_checkpoints_for_each_welfare_in_dir(data_dir, hp):
    all_replicates_save_dir = helpers.get_dir_of_each_replicate(
        data_dir, str_in_dir="LOLAExactTrainer_"
    )
    assert len(all_replicates_save_dir) > 0
    welfares = _classify_base_replicates_into_welfares(all_replicates_save_dir)

    ckpt_per_welfare = {}
    for welfare_fn, welfare_name in hp["welfare_functions"]:
        replicates_save_dir_for_welfare = _filter_replicate_dir_by_welfare(
            all_replicates_save_dir, welfares, welfare_name
        )
        ckpts = restore.get_checkpoint_for_each_replicates(
            replicates_save_dir_for_welfare
        )
        ckpt_per_welfare[welfare_name] = [ckpt + ".json" for ckpt in ckpts]
    return ckpt_per_welfare


def _classify_base_replicates_into_welfares(all_replicates_save_dir):
    welfares = []
    for replicate_dir in all_replicates_save_dir:
        reward_player_1, reward_player_2 = _get_last_episode_rewards(
            replicate_dir
        )
        welfare_name = (
            lola_exact_official.classify_into_welfare_based_on_rewards(
                reward_player_1, reward_player_2
            )
        )
        welfares.append(welfare_name)
    return welfares


def _filter_replicate_dir_by_welfare(
    all_replicates_save_dir, welfares, welfare_name
):
    replicates_save_dir_for_welfare = [
        replicate_dir
        for welfare, replicate_dir in zip(welfares, all_replicates_save_dir)
        if welfare == welfare_name
    ]
    return replicates_save_dir_for_welfare


def _get_last_episode_rewards(replicate_dir):
    results = utils.path.get_results_for_replicate(replicate_dir)
    last_epsiode_results = results[-1]
    return last_epsiode_results["ret1"], last_epsiode_results["ret2"]


def _get_all_welfare_pairs_wt_cross_play_payoffs(hp, player_ids):
    raw_data_points_wt_welfares = {}
    eval_results_path = _get_path_to_eval_results(hp)
    with open(eval_results_path) as json_file:
        json_content = json.load(json_file)
    for mode, results in json_content.items():
        if "self-play" in mode:
            continue
        raw_players_perf = [None, None]
        for metric, metric_result in results.items():
            if "player_row" in metric:
                raw_players_perf[0] = metric_result["mean"]
            elif "player_col" in metric:
                raw_players_perf[1] = metric_result["mean"]
            else:
                raise ValueError()
        play_mode = mode.split(":")[-1]
        play_mode = play_mode.replace(" vs ", "-")
        play_mode = play_mode.strip()
        raw_data_points_wt_welfares[play_mode] = raw_players_perf

    all_welfare_pairs_wt_payoffs = _adjust_perf_for_epi_length(
        raw_data_points_wt_welfares, hp
    )
    print("all_welfare_pairs_wt_payoffs", all_welfare_pairs_wt_payoffs)
    return all_welfare_pairs_wt_payoffs


def _get_path_to_eval_results(hp):
    child_dirs = utils.path.get_children_paths_wt_discarding_filter(
        hp["data_dir"], _filter="LOLAExact"
    )
    child_dirs = utils.path.keep_dirs_only(child_dirs)
    assert len(child_dirs) == 1, f"{child_dirs}"
    eval_dir = utils.path.get_unique_child_dir(child_dirs[0])
    possible_nested_dir = utils.path.try_get_unique_child_dir(eval_dir)
    if possible_nested_dir is not None:
        eval_dir = possible_nested_dir
    eval_dir = os.path.join(
        eval_dir,
        "plotself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json",
    )
    return eval_dir


def _adjust_perf_for_epi_length(raw_data_points_wt_welfares, hp):
    all_welfare_pairs_wt_payoffs = {}
    for (
        play_mode,
        values_per_players,
    ) in raw_data_points_wt_welfares.items():
        all_welfare_pairs_wt_payoffs[play_mode] = (
            values_per_players[0] / hp["trace_length"],
            values_per_players[1] / hp["trace_length"],
        )
    return all_welfare_pairs_wt_payoffs


def _get_rllib_config_for_base_lola_exact_policy(hp):
    lola_exact_hp = lola_exact_official.get_hyperparameters(
        debug=hp["debug"], env="IteratedAsymBoS", train_n_replicates=1
    )
    (
        hp_eval,
        rllib_config,
        policies_to_load,
        trainable_class,
        stop_config,
        env_config,
    ) = lola_exact_official.generate_eval_config(lola_exact_hp)

    trainer = PGTrainer

    return stop_config, env_config, rllib_config, trainer, lola_exact_hp


# def _extract_checkpoints_used_for_each_players(
#     player_ids, eval_replicate_path
# ):
#     params = utils.path.get_params_for_replicate(eval_replicate_path)
#     policies_config = params["multiagent"]["policies"]
#     ckps = [
#         policies_config[player_id][3]["checkpoint_to_load_from"][0]
#         for player_id in player_ids
#     ]
#     return ckps
#
#
# def _is_cross_play(players_ckpts):
#     return players_ckpts[0] != players_ckpts[1]


# def _convert_checkpoint_names_to_welfares(hp, players_ckpts):
#     players_welfares = []
#     for player_ckpt in players_ckpts:
#         player_ckpt_wtout_root = "/".join(player_ckpt.split("/")[-4:])
#         for welfare, ckpts_for_welfare in hp["ckpt_per_welfare"].items():
#             if any(
#                 player_ckpt_wtout_root in ckpt for ckpt in ckpts_for_welfare
#             ):
#                 players_welfares.append(welfare)
#                 break
#
#     assert len(players_welfares) == len(
#         players_ckpts
#     ), f"{len(players_welfares)} == {len(players_ckpts)}"
#     return players_welfares


# def _extract_performance(eval_replicate_path, player_ids):
#     results_per_epi = utils.path.get_results_for_replicate(eval_replicate_path)
#     players_avg_reward = _extract_and_average_perf(results_per_epi, player_ids)
#     return players_avg_reward
#
#
# def _extract_and_average_perf(results_per_epi, player_ids):
#     players_avg_reward = []
#     for player_id in player_ids:
#         player_rewards = []
#         for result_in_one_epi in results_per_epi:
#             total_player_reward_in_one_epi = result_in_one_epi[
#                 "policy_reward_mean"
#             ][player_id]
#             player_rewards.append(total_player_reward_in_one_epi)
#         players_avg_reward.append(sum(player_rewards) / len(player_rewards))
#     return players_avg_reward
#
#
# def _get_play_mode(players_welfares):
#     return f"{players_welfares[0]}-{players_welfares[1]}"


# def _average_perf_per_play_mode(raw_data_points_wt_welfares, hp):
#     all_welfare_pairs_wt_payoffs = {}
#     for (
#         play_mode,
#         values_per_replicates,
#     ) in raw_data_points_wt_welfares.items():
#         player_1_values = [
#             value_replicate[0] for value_replicate in values_per_replicates
#         ]
#         player_2_values = [
#             value_replicate[1] for value_replicate in values_per_replicates
#         ]
#         all_welfare_pairs_wt_payoffs[play_mode] = (
#             sum(player_1_values) / len(player_1_values) / hp["trace_length"],
#             sum(player_2_values) / len(player_2_values) / hp["trace_length"],
#         )
#     return all_welfare_pairs_wt_payoffs


#
# def _get_list_of_replicates_path_in_eval(hp):
#     child_dirs = utils.path.get_children_paths_wt_discarding_filter(
#         hp["data_dir"], _filter="LOLAExact"
#     )
#     child_dirs = utils.path.keep_dirs_only(child_dirs)
#     assert len(child_dirs) == 1, f"{child_dirs}"
#     eval_dir = utils.path.get_unique_child_dir(child_dirs[0])
#     eval_replicates_dir = utils.path.get_unique_child_dir(eval_dir)
#     possible_nested_dir = utils.path.try_get_unique_child_dir(
#         eval_replicates_dir
#     )
#     if possible_nested_dir is not None:
#         eval_replicates_dir = possible_nested_dir
#     all_eval_replicates_dirs = (
#         utils.path.get_children_paths_wt_selecting_filter(
#             eval_replicates_dir, _filter="PG_"
#         )
#     )
#     return all_eval_replicates_dirs


# def _get_all_welfare_pairs_wt_cross_play_payoffs(hp, player_ids):
# all_eval_replicates_dirs = _get_list_of_replicates_path_in_eval(hp)
# raw_data_points_wt_welfares = {}
# for eval_replicate_path in all_eval_replicates_dirs:
#     players_ckpts = _extract_checkpoints_used_for_each_players(
#         player_ids, eval_replicate_path
#     )
#     if _is_cross_play(players_ckpts):
#         players_welfares = _convert_checkpoint_names_to_welfares(
#             hp, players_ckpts
#         )
#         raw_players_perf = _extract_performance(
#             eval_replicate_path, player_ids
#         )
#         play_mode = _get_play_mode(players_welfares)
#         if play_mode not in raw_data_points_wt_welfares.keys():
#             raw_data_points_wt_welfares[play_mode] = []
#         raw_data_points_wt_welfares[play_mode].append(raw_players_perf)
# all_welfare_pairs_wt_payoffs = _average_perf_per_play_mode(
#     raw_data_points_wt_welfares, hp
# )
