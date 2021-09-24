import numpy as np

from submission.plots.normalize_score_meta_solvers import (
    _compute_metric_4,
    welfare_functions,
    welfare_disagreements_payoffs,
    welfare_maximums,
)
from submission.utils.path import _read_json_file
from submission.plots.plot_bar_chart_from_saved_results import (
    EMPIRICAL_WELFARE_OPTIMUM_CG,
    EMPIRICAL_WELFARE_OPTIMUM_ABCG,
)


# env = "CG"
env = "ABCG"
# env = "AsymBoS"
if env == "CG":
    path_to_file = "/home/maxime/dev-maxime/CLR/vm-data/instance-20-cpu-1-memory-x2/amTFT/2021_09_25/18_04_34/eval/2021_09_25/18_04_39/plotself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json"
    episode_length = 100
    # welfare_disagreements_payoffs = (0.0, 0.0)
    # welfare_maximums = [
    #     el(EMPIRICAL_WELFARE_OPTIMUM_CG) for el in welfare_functions
    # ]
    print("welfare_maximums", welfare_maximums)
    metric_p1 = "Metric:policy_reward_mean/player_red, Metric mode:avg"
    metric_p2 = "Metric:policy_reward_mean/player_blue, Metric mode:avg"
elif env == "ABCG":
    path_to_file = "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-2/amTFT/2021_09_25/20_10_35/eval/2021_09_25/20_10_42/plotself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json"
    path_to_file = (
        "/home/maxime/dev-maxime/CLR/vm-data/instance-60-cpu-1-preemtible/amTFT"
        "/2021_09_25/21_35_10/eval/2021_09_25/21_35_15/plotself_and_cross_play_policy_reward_mean_player_blue_vs_policy_reward_mean_player_red_matrix.json"
    )
    episode_length = 100
    # welfare_disagreements_payoffs = (0.0, 0.0)
    # welfare_maximums = [
    #     el(EMPIRICAL_WELFARE_OPTIMUM_ABCG) for el in welfare_functions
    # ]
    metric_p1 = "Metric:policy_reward_mean/player_red, Metric mode:avg"
    metric_p2 = "Metric:policy_reward_mean/player_blue, Metric mode:avg"
elif env == "AsymBoS":
    episode_length = 20
    path_to_file = (
        "/home/maxime/dev-maxime/CLR/vm-data/instance-60-cpu-1"
        "-preemtible/amTFT/2021_09_23/12_50_52/eval/2021_09_23/12_51_02/plotself_and_cross_play_policy_reward_mean_player_col_vs_policy_reward_mean_player_row_matrix.json"
    )
    metric_p1 = "Metric:policy_reward_mean/player_row, Metric mode:avg"
    metric_p2 = "Metric:policy_reward_mean/player_col, Metric mode:avg"


def main(debug):
    data = _read_json_file(path_to_file)

    all_play_mode_values = []
    for play_mode, play_mode_values in data.items():
        play_mode_values = get_payoffs(play_mode_values)
        all_play_mode_values.append(play_mode_values)
    all_play_mode_values = np.concatenate(all_play_mode_values, axis=0)

    results = {}
    for play_mode, play_mode_values in data.items():
        play_mode_values = get_payoffs(play_mode_values)

        print("play_mode_values", play_mode_values.shape, play_mode_values)
        normalized_score = _compute_metric_4(
            play_mode_values,
            welfare_functions,
            welfare_disagreements_payoffs,
            welfare_disagreements_payoffs,
            search_max=True,
            payoffs_for_all_solvers=all_play_mode_values,
        )
        print(play_mode, "normalized_score", normalized_score)
        results[play_mode] = normalized_score
    print(results)


def get_payoffs(play_mode_values):
    play_mode_values_player_2 = play_mode_values[metric_p2]["raw_data"]
    play_mode_values_player_1 = play_mode_values[metric_p1]["raw_data"]
    play_mode_values_player_1 = convert_to_list(play_mode_values_player_1)
    play_mode_values_player_2 = convert_to_list(play_mode_values_player_2)
    play_mode_values = np.array(
        [play_mode_values_player_1, play_mode_values_player_2]
    )
    play_mode_values = np.transpose(play_mode_values, (1, 0))
    play_mode_values = play_mode_values / episode_length
    return play_mode_values


def convert_to_list(str_):
    list_str_ = (
        str_.replace("[", "").replace("]", "").replace(" ", "").split(",")
    )
    return [float(el) for el in list_str_]


if __name__ == "__main__":
    debug = False
    main(debug)
