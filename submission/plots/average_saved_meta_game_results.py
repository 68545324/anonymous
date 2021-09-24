import json
import os

import numpy as np
from submission.meta_game_experiments.meta_game import (
    META_MINIMUM,
    META_UNIFORM,
    META_RANDOM,
    META_APLHA_RANK_MIX,
    META_APLHA_RANK_MAX,
    META_REPLICATOR_DYNAMIC,
)

OLD_FORMAT = "13_50_55"

# Meta policies provided
PREFIX = os.path.join(
    os.path.dirname(__file__),
    "../meta_game_experiments/results/meta_game_compare",
)
# New meta policies
# PREFIX = "~/ray_results/meta_game_compare"

META_POLICY_SAVE_PATHS = {
    META_MINIMUM: os.path.join(
        PREFIX, "2021_05_27/19_24_36/meta_game/meta_policies.json"
    ),
    META_UNIFORM: os.path.join(
        PREFIX, "2021_05_14/13_50_55/meta_game/meta_policies.json"
    ),
    META_APLHA_RANK_MIX: os.path.join(
        PREFIX, "2021_05_14/10_37_24/meta_game/meta_policies.json"
    ),
    META_APLHA_RANK_MAX: os.path.join(
        PREFIX, "2021_05_14/10_39_47/meta_game/meta_policies.json"
    ),
    META_REPLICATOR_DYNAMIC: os.path.join(
        PREFIX, "2021_05_14/10_42_10/meta_game/meta_policies.json"
    ),
    META_RANDOM: os.path.join(
        PREFIX, "2021_05_14/10_50_36/meta_game/meta_policies.json"
    ),
}


def main(debug):
    prefix, files_data, n_players = _get_inputs()
    files_to_process = _preprocess_inputs(prefix, files_data)

    for file, file_data in zip(files_to_process, files_data):
        (
            mean_per_player,
            std_dev_per_player,
            std_err_per_player,
            coordination_success,
        ) = _get_stats_for_file(file, n_players, file_data)

        print(
            file_data[0],
            "mean:",
            mean_per_player,
            "std_dev:",
            std_dev_per_player,
            "std_err:",
            std_err_per_player,
            "mean coordination_success:",
            coordination_success,
        )


def _get_inputs():
    prefix = ""
    EPISODE_LENGTH = 200

    files_data = [
        (
            k,
            EPISODE_LENGTH,
            os.path.join(
                os.path.split(v)[0],
                "final_base_game/final_eval_in_base_game.json",
            ),
        )
        if OLD_FORMAT not in v
        else (
            k,
            EPISODE_LENGTH,
            os.path.join(
                os.path.split(os.path.split(v)[0])[0],
                "final_eval_in_base_game.json",
            ),
        )
        for k, v in META_POLICY_SAVE_PATHS.items()
    ]
    n_players = 2
    return prefix, files_data, n_players


def _preprocess_inputs(prefix, files_data):
    files_to_process = [
        os.path.join(prefix, file_data[2]) for file_data in files_data
    ]
    return files_to_process


def _get_stats_for_file(file, n_players, file_data):
    values_per_replicat_per_player, coordination_success = _raw_data_for_file(
        file, n_players, file_data
    )

    n_replicates_in_content = values_per_replicat_per_player.shape[0]

    mean_per_player = values_per_replicat_per_player.mean(axis=0)
    std_dev_per_player = values_per_replicat_per_player.std(axis=0)
    std_err_per_player = std_dev_per_player / np.sqrt(n_replicates_in_content)
    return (
        mean_per_player,
        std_dev_per_player,
        std_err_per_player,
        coordination_success,
    )


def _raw_data_for_file(file, n_players, file_data):
    file_path = os.path.expanduser(file)
    with (open(file_path, "rb")) as f:
        file_content = json.load(f)
        file_content = _format_2nd_into_1st_format(file_content, file_data)
        if isinstance(file_content, dict):
            coordination_success = file_content["mean_coordination_success"]
            file_content = file_content["results"]
        else:
            coordination_success = "N.A."

        if OLD_FORMAT in file_path:
            file_content = file_content[0][2]
            values_per_replicat_per_player = np.array(file_content)
            values_per_replicat_per_player = (
                values_per_replicat_per_player.transpose(1, 0)
            )
        else:
            values_per_replicat_per_player = np.array(file_content)

        assert values_per_replicat_per_player.ndim == 2
        n_players_in_content = values_per_replicat_per_player.shape[1]
        assert n_players_in_content == n_players

    values_per_replicat_per_player = (
        values_per_replicat_per_player / file_data[1]
    )
    return values_per_replicat_per_player, coordination_success


def _format_2nd_into_1st_format(file_content, file_data):
    if len(file_data) == 4:
        file_content = file_content[0][2]
        new_format = []
        for p1_content, p2_content in zip(file_content[0], file_content[1]):
            new_format.append((p1_content, p2_content))
        file_content = new_format
    return file_content


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
