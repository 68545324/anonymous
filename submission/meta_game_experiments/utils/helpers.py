import json
import os

from submission.utils import plot, cross_play, path


def save_to_json(exp_name, object, filename="final_eval_in_base_game.json"):
    exp_dir = get_exp_dir_from_exp_name(exp_name)
    json_file = os.path.join(exp_dir, filename)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    with open(json_file, "w") as outfile:
        json.dump(object, outfile)


def get_exp_dir_from_exp_name(exp_name: str):
    """
    :param exp_name: exp_name provided to tune.run
    :return: path to the experiment analysis repository (ray log dir)
    """
    exp_dir = os.path.join("~/ray_results", exp_name)
    exp_dir = os.path.expanduser(exp_dir)
    return exp_dir


def plot_results(
    exp_name, results, hp_eval, format_fn, jitter=0.0, title=None
):
    exp_dir = get_exp_dir_from_exp_name(exp_name)
    data_groups_per_mode = format_fn(results)

    background_area_coord = None
    if "env_class" in hp_eval.keys():
        background_area_coord = hp_eval["env_class"].PAYOFF_MATRIX

    plot_config = plot.PlotConfig(
        title=title,
        save_dir_path=exp_dir,
        xlim=hp_eval["x_limits"],
        ylim=hp_eval["y_limits"],
        markersize=5,
        jitter=jitter,
        xlabel="player 1 payoffs",
        ylabel="player 2 payoffs",
        x_scale_multiplier=hp_eval["plot_axis_scale_multipliers"][0],
        y_scale_multiplier=hp_eval["plot_axis_scale_multipliers"][1],
        background_area_coord=background_area_coord,
    )

    plot_helper = plot.PlotHelper(plot_config)
    plot_helper.plot_dots(data_groups_per_mode)


def _mix_rllib_config(all_rllib_configs, hp_eval):
    if (
        hp_eval["n_self_play_in_final_meta_game"] == 1
        and hp_eval["n_cross_play_in_final_meta_game"] == 0
    ):
        return all_rllib_configs[0]
    elif (
        hp_eval["n_self_play_in_final_meta_game"] == 0
        and hp_eval["n_cross_play_in_final_meta_game"] != 0
    ):
        master_config = cross_play.utils.mix_policies_in_given_rllib_configs(
            all_rllib_configs,
            n_mix_per_config=hp_eval["n_cross_play_in_final_meta_game"],
        )
        return master_config
    else:
        raise ValueError()


def get_dir_of_each_replicate(welfare_training_save_dir, str_in_dir="DQN_"):
    return path.get_children_paths_wt_selecting_filter(
        welfare_training_save_dir, _filter=str_in_dir
    )
