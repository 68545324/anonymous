import copy
import logging
import os

import ray
from ray.rllib.agents import dqn
import numpy as np
from submission import utils
from submission.algos import (
    amTFT,
    multi_path_cooperator,
    hierarchical,
)
from submission.base_game_experiments.amtft_various_env import (
    get_hyperparameters,
    load_experiment_analysis,
    _generate_eval_config,
    get_rllib_config,
    plot_evaluation,
)
from submission.utils import path as path_util
from submission.utils import (
    postprocessing,
    restore,
    cross_play,
)
from matplotlib import pyplot as plt
from submission.plots.normalize_score_meta_solvers import _compute_metric_4

logger = logging.getLogger(__name__)

use_min_punish_steps = False
min_punish_steps = 10

use_modified_payoff_matrix = False
# CAUTION: must modify the env code to set back the payoff matrix

use_random_switching = True


epi_length = 20
use_simple_eval = True


def main(
    debug,
    train_n_replicates=None,
    filter_utilitarian=None,
    env=None,
    use_r2d2=False,
):
    if use_simple_eval:
        preferred_w_probabilities = [0.5]
    elif debug:
        preferred_w_probabilities = [0.5, 0.75]
    else:
        preferred_w_probabilities = np.round_(
            np.arange(0.5, 1.0 + 0.01, 0.1).tolist(), 2
        )

    results_per_preferred_proba = []
    for preferred_w_proba in preferred_w_probabilities:
        # Try this between 1 and 3 times
        n_failures = 0
        while n_failures < 3:
            # try:
            ray.shutdown()
            (
                analysis_metrics_per_mode,
                evaluator,
                hparams,
                experiment_analysis_per_welfare,
            ) = _one_main(
                debug,
                train_n_replicates,
                filter_utilitarian,
                env,
                preferred_w_proba,
                use_r2d2,
            )
            # except Exception as e:
            #     n_failures += 1
            #     if n_failures < 3:
            #         continue
            #     else:
            #         raise e
            break

        metrics = (
            np.array(
                np.array(
                    evaluator.plotter.data_groups_per_mode["cross-play"]
                ).tolist()
            ).squeeze(axis=1)
            / epi_length
        )
        normalized_score = _compute_metric_4(
            metrics,
        )
        results_per_preferred_proba.append(normalized_score)

    _plot_robustness_exploitability_tradeoff(
        results_per_preferred_proba, hparams, preferred_w_probabilities
    )

    return experiment_analysis_per_welfare, analysis_metrics_per_mode


def _one_main(
    debug,
    train_n_replicates,
    filter_utilitarian,
    env,
    preferred_w_proba,
    use_r2d2,
):
    hparams = get_hyperparameters(
        debug,
        train_n_replicates,
        filter_utilitarian,
        env,
        use_r2d2=use_r2d2,
    )

    hparams = _modify_hp_for_meta_amTFT(hparams, preferred_w_proba)

    if hparams["env_name"] in ["IteratedAsymBoS", "IteratedAsymBoSandPD"]:
        hparams = _filter_utilitarian_results_with_inequity_aversion_outcomes(
            hparams
        )

    ray.init(
        num_gpus=0,
        num_cpus=os.cpu_count(),
        local_mode=hparams["debug"],
    )

    # Train
    if hparams["load_policy_data"] is None:
        raise NotImplementedError(
            "You need to train the agents using the amtft_various_env.py "
            "script and then load their checkpoints. You must provide your "
            "own paths inside the _get_saved_base_amTFT_checkpoint_paths "
            "function."
        )
    else:
        experiment_analysis_per_welfare = load_experiment_analysis(
            hparams["load_policy_data"]
        )

    # Eval & Plot
    analysis_metrics_per_mode, evaluator = config_and_evaluate_cross_play(
        experiment_analysis_per_welfare, hparams
    )

    ray.shutdown()
    return (
        analysis_metrics_per_mode,
        evaluator,
        hparams,
        experiment_analysis_per_welfare,
    )


def _plot_robustness_exploitability_tradeoff(
    results_per_preferred_proba, hparams, preferred_w_probabilities
):
    print("results_per_preferred_proba", results_per_preferred_proba)
    plt.plot(preferred_w_probabilities, results_per_preferred_proba)
    plt.xlabel("Selection probability of prefered welfare")
    plt.ylabel("Avg normalized score in cross play")
    base_dir = os.getenv("TUNE_RESULT_DIR", "~/ray_results")
    base_dir = os.path.expanduser(base_dir)
    save_path = os.path.join(
        base_dir,
        hparams["exp_name"],
        f"robustness_exploitability_tradeoff_{hparams['env_name']}.png",
    )
    plt.savefig(save_path)


def _modify_hp_for_meta_amTFT(hp, preferred_w_proba):
    if hp["debug"]:
        hp["train_n_replicates"] = 1
        # hp["n_self_play_per_checkpoint"] = 1
        hp["n_cross_play_per_checkpoint"] = 1

    if hp["env_name"] == "IteratedAsymBoS":
        hp["x_limits"] = (-0.6, 4.1)
        hp["y_limits"] = (-0.1, 4.1)
        if use_modified_payoff_matrix:
            hp["x_limits"] = (-0.1, 8.1)
            hp["y_limits"] = (-2.1, 4.1)
    elif hp["env_name"] == "IteratedAsymBoSandPD":
        assert not use_modified_payoff_matrix
        hp["x_limits"] = (-5.1, 4.1)
        hp["y_limits"] = (-3.1, 6.1)
    elif hp["env_name"] in ["CoinGame", "ABCoinGame"]:
        pass
    else:
        raise ValueError()

    (
        prefix,
        util_load_data_list,
        ia_load_data_list,
    ) = _get_saved_base_amTFT_checkpoint_paths(hp["env_name"])

    ia_load_data_list = ia_load_data_list[: hp["train_n_replicates"]]
    if hp["env_name"] in ["IteratedAsymBoS", "IteratedAsymBoSandPD"]:
        util_load_data_list = util_load_data_list[
            : int(hp["train_n_replicates"] * 4)
        ]
    else:
        util_load_data_list = util_load_data_list[
            : int(hp["train_n_replicates"])
        ]

    util_prefix = os.path.join(prefix, "utilitarian_welfare/coop")
    inequity_aversion_prefix = os.path.join(
        prefix, "inequity_aversion_welfare/coop"
    )

    hp["load_policy_data"] = {
        "Util": [
            os.path.join(util_prefix, path) for path in util_load_data_list
        ],
        "IA": [
            os.path.join(inequity_aversion_prefix, path)
            for path in ia_load_data_list
        ],
    }

    hp[multi_path_cooperator.PREFERRED_W_PROBA_KEY] = preferred_w_proba

    return hp


def _get_saved_base_amTFT_checkpoint_paths(env_name):
    if env_name == "IteratedAsymBoS":
        prefix = (
            "/home/maxime/dev-maxime/CLR/vm-data/instance-60-cpu-1-preemtible/"
            "amTFT/2021_05_11/07_40_04"
        )
        # Prefix to use inside VM
        # prefix = (
        #     os.path.expanduser("~/ray_results/amTFT/2021_05_11/07_40_04")
        # )
        util_load_data_list = [
            "R2D2_IteratedAsymBoS_9463a_00000_0_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-50/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00001_1_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00002_2_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00003_3_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00004_4_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00005_5_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00006_6_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00007_7_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00008_8_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00009_9_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00010_10_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-13-56/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00011_11_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-18-53/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00012_12_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-18-53/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00013_13_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-19-13/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00014_14_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-19-13/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00015_15_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-19-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00016_16_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-19-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00017_17_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-19-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00018_18_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-19-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00019_19_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-19-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00020_20_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-19-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00021_21_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-24-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00022_22_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-24-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00023_23_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-24-18/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00024_24_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-24-19/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00025_25_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-24-33/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00026_26_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-24-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00027_27_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-24-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00028_28_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-24-59/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00029_29_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-25-00/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00030_30_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-25-00/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00031_31_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-29-22/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00032_32_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-29-32/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00033_33_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-29-32/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00034_34_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00035_35_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-29-58/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00036_36_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-29-59/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00037_37_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-30-24/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00038_38_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-30-24/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00039_39_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-30-45/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00040_40_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-30-45/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00041_41_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-34-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00042_42_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-35-12/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00043_43_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-35-13/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00044_44_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-35-32/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00045_45_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-35-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00046_46_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-35-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00047_47_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-35-53/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00048_48_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-35-53/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00049_49_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-36-04/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00050_50_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-36-15/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00051_51_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-40-02/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00052_52_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-40-18/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00053_53_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-40-23/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00054_54_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-40-39/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00055_55_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-41-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00056_56_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-41-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00057_57_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-41-14/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00058_58_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-41-14/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00059_59_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-41-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00060_60_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-41-46/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00061_61_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-45-32/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00062_62_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-45-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00063_63_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-45-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00064_64_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-46-07/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00065_65_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-46-07/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00066_66_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-46-27/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00067_67_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-46-28/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00068_68_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-46-44/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00069_69_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-46-51/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00070_70_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-47-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00071_71_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-50-45/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00072_72_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-51-05/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00073_73_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-51-25/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00074_74_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-51-26/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00075_75_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-51-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00076_76_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-52-02/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00077_77_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-52-07/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00078_78_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-52-08/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00079_79_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-52-23/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00080_80_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-52-33/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00081_81_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-55-54/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00082_82_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-56-26/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00083_83_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-56-42/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00084_84_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-56-53/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00085_85_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-57-18/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00086_86_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-57-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00087_87_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-57-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00088_88_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-57-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00089_89_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-57-50/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00090_90_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_10-58-00/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00091_91_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-01-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00092_92_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-01-50/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00093_93_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-02-11/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00094_94_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-02-22/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00095_95_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-02-59/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00096_96_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-03-06/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00097_97_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-03-07/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00098_98_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-03-07/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00099_99_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_11-03-22/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00100_100_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-03-32/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00101_101_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-07-01/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00102_102_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-07-27/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00103_103_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-07-42/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00104_104_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-07-52/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00105_105_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-08-13/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00106_106_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-08-13/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00107_107_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-08-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00108_108_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-08-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00109_109_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-08-50/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00110_110_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-09-11/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00111_111_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-12-34/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00112_112_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-12-50/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00113_113_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-13-16/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00114_114_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-13-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00115_115_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-13-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00116_116_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-13-50/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00117_117_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-14-01/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00118_118_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-14-12/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00119_119_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-14-32/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00120_120_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-14-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00121_121_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-18-05/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00122_122_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-18-05/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00123_123_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-18-53/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00124_124_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-19-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00125_125_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-19-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00126_126_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-19-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00127_127_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-19-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00128_128_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-19-49/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00129_129_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-20-00/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00130_130_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-20-26/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00131_131_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-23-42/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00132_132_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-23-42/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00133_133_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-24-18/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00134_134_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-24-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00135_135_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-24-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00136_136_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-24-46/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00137_137_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-24-46/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00138_138_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-25-27/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00139_139_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-25-27/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00140_140_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-25-43/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00141_141_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-28-58/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00142_142_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-29-45/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00143_143_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-29-45/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00144_144_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-29-50/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00145_145_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-29-50/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00146_146_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-30-01/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00147_147_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-30-17/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00148_148_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-30-38/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00149_149_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-30-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00150_150_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-30-59/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00151_151_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-34-05/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00152_152_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-34-46/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00153_153_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-34-46/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00154_154_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-35-01/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00155_155_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-35-33/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00156_156_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-35-33/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00157_157_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-36-17/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00158_158_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-36-17/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_9463a_00159_159_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecewise_2021-05-11_11-36-18/checkpoint_000050/checkpoint-50",
        ]
        ia_load_data_list = [
            "R2D2_IteratedAsymBoS_f74e8_00000_0_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-45/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00001_1_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00002_2_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00003_3_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00004_4_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00005_5_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00006_6_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00007_7_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00008_8_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00009_9_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSc_2021-05-11_08-07-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00010_10_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-07-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00011_11_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-15-44/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00012_12_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-15-44/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00013_13_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-15-44/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00014_14_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-15-44/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00015_15_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-15-44/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00016_16_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-16-15/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00017_17_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-16-15/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00018_18_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-16-16/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00019_19_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-16-16/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00020_20_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-16-16/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00021_21_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-23-21/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00022_22_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-23-52/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00023_23_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-23-52/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00024_24_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-24-02/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00025_25_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-24-02/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00026_26_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-24-02/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00027_27_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-24-02/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00028_28_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-24-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00029_29_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-24-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00030_30_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-24-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00031_31_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-31-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00032_32_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-31-18/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00033_33_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-31-41/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00034_34_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-31-41/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00035_35_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-32-10/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00036_36_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-32-10/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00037_37_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-32-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00038_38_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-32-46/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoS_f74e8_00039_39_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseS_2021-05-11_08-32-46/checkpoint_000050/checkpoint-50",
        ]
    elif env_name == "IteratedAsymBoSandPD":
        prefix = (
            "/home/maxime/dev-maxime/CLR/vm-data/instance-60-cpu-1-preemtible/"
            "amTFT/2021_09_23/18_23_20"
        )
        # Prefix to use inside VM
        # prefix = os.path.expanduser("~/ray_results/amTFT/2021_09_23/18_23_20")
        util_load_data_list = [
            "R2D2_IteratedAsymBoSandPD_610f4_00000_0_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-30/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00001_1_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00002_2_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00003_3_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00004_4_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00005_5_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00006_6_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00007_7_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00008_8_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00009_9_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00010_10_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00011_11_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00012_12_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00013_13_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00014_14_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00015_15_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00016_16_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00017_17_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00018_18_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00019_19_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00020_20_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-35/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00021_21_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00022_22_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00023_23_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00024_24_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00025_25_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00026_26_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00027_27_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00028_28_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00029_29_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00030_30_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00031_31_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00032_32_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00033_33_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00034_34_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00035_35_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00036_36_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00037_37_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00038_38_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00039_39_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00040_40_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00041_41_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00042_42_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00043_43_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00044_44_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00045_45_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00046_46_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00047_47_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00048_48_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00049_49_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00050_50_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00051_51_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00052_52_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00053_53_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00054_54_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00055_55_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00056_56_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00057_57_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00058_58_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00059_59_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00060_60_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-59-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00061_61_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-12/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00062_62_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-13/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00063_63_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-14/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00064_64_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-16/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00065_65_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-16/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00066_66_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-17/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00067_67_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-17/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00068_68_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-17/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00069_69_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-18/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00070_70_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-18/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00071_71_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-19/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00072_72_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-19/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00073_73_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-19/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00074_74_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-19/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00075_75_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-20/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00076_76_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-20/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00077_77_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-21/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00078_78_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-21/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00079_79_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-21/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00080_80_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-22/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00081_81_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-22/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00082_82_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-22/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00083_83_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-23/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00084_84_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-23/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00085_85_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-23/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00086_86_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-23/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00087_87_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-23/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00088_88_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-23/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00089_89_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-24/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00090_90_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-24/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00091_91_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-24/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00092_92_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-25/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00093_93_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-25/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00094_94_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-25/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00095_95_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-25/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00096_96_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-26/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00097_97_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-26/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00098_98_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-26/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00099_99_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_19-07-26/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00100_100_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-27/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00101_101_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-27/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00102_102_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-27/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00103_103_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-27/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00104_104_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-28/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00105_105_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-28/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00106_106_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-28/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00107_107_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00108_108_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00109_109_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-29/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00110_110_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-30/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00111_111_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-31/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00112_112_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-31/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00113_113_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-31/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00114_114_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-33/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00115_115_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-33/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00116_116_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-36/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00117_117_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-37/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00118_118_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-40/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00119_119_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-41/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00120_120_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-07-45/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00121_121_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-02/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00122_122_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00123_123_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-03/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00124_124_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-05/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00125_125_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-05/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00126_126_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-05/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00127_127_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-06/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00128_128_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-06/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00129_129_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-06/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00130_130_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-07/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00131_131_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-08/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00132_132_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-08/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00133_133_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-09/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00134_134_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-10/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00135_135_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-10/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00136_136_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-10/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00137_137_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-11/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00138_138_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-11/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00139_139_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-11/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00140_140_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-11/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00141_141_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-12/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00142_142_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-12/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00143_143_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-12/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00144_144_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-12/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00145_145_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-12/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00146_146_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-13/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00147_147_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-13/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00148_148_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-14/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00149_149_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-14/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00150_150_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-14/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00151_151_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-15/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00152_152_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-15/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00153_153_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-15/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00154_154_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-15/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00155_155_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-15/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00156_156_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-16/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00157_157_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-16/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00158_158_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-17/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_610f4_00159_159_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piec_2021-09-23_19-15-17/checkpoint_000050/checkpoint-50",
        ]
        ia_load_data_list = [
            "R2D2_IteratedAsymBoSandPD_3a0c4_00000_0_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-46/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00001_1_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00002_2_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00003_3_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00004_4_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00005_5_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00006_6_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00007_7_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00008_8_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00009_9_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piecew_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00010_10_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00011_11_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00012_12_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00013_13_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00014_14_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00015_15_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00016_16_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00017_17_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00018_18_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00019_19_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00020_20_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00021_21_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00022_22_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00023_23_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00024_24_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00025_25_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00026_26_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-47/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00027_27_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00028_28_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00029_29_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00030_30_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00031_31_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00032_32_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00033_33_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00034_34_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00035_35_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00036_36_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00037_37_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00038_38_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
            "R2D2_IteratedAsymBoSandPD_3a0c4_00039_39_buffer_size=2000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Piece_2021-09-23_18-29-48/checkpoint_000050/checkpoint-50",
        ]
    elif env_name == "CoinGame":
        prefix = (
            "/home/maxime/dev-maxime/CLR/vm-data/instance-20-cpu-1-memory-x2/"
            "amTFT/2021_05_15/07_16_37"
        )
        # Prefix to use inside VM
        # prefix = (
        #     os.path.expanduser("~/ray_results/amTFT/2021_05_15/07_16_37")
        # )
        util_load_data_list = [
            "R2D2_CoinGame_0dc16_00000_0_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-23/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00001_1_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-24/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00002_2_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-24/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00003_3_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00004_4_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00005_5_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00006_6_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00007_7_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00008_8_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00009_9_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00010_10_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00011_11_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00012_12_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-26/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00013_13_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-26/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00014_14_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-26/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00015_15_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-26/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00016_16_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-27/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00017_17_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-27/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00018_18_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-27/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00019_19_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-27/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00020_20_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_01-57-27/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00021_21_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_04-59-53/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00022_22_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-00-01/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00023_23_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-00-13/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00024_24_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-00-36/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00025_25_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-00-36/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00026_26_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-00-36/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00027_27_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-00-46/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00028_28_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-00-56/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00029_29_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-01/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00030_30_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-11/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00031_31_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-17/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00032_32_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-17/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00033_33_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-22/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00034_34_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-23/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00035_35_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-24/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00036_36_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-29/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00037_37_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-30/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00038_38_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-43/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_0dc16_00039_39_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-16_05-01-51/checkpoint_000500/checkpoint-500",
        ]
        ia_load_data_list = [
            "R2D2_CoinGame_9f3d3_00000_0_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-49/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00001_1_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-51/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00002_2_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-51/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00003_3_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-51/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00004_4_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00005_5_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00006_6_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00007_7_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00008_8_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00009_9_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedul_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00010_10_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00011_11_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00012_12_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00013_13_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-53/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00014_14_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-53/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00015_15_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-53/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00016_16_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-53/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00017_17_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-53/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00018_18_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-54/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00019_19_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-54/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00020_20_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_13-29-54/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00021_21_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-31-21/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00022_22_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-31-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00023_23_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-32-09/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00024_24_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-32-23/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00025_25_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-32-29/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00026_26_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-32-52/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00027_27_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-32-57/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00028_28_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-32-57/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00029_29_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-32-58/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00030_30_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-33-09/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00031_31_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-33-09/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00032_32_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-33-34/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00033_33_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-33-35/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00034_34_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-33-35/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00035_35_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-33-47/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00036_36_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-33-47/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00037_37_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-33-54/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00038_38_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-34-25/checkpoint_000500/checkpoint-500",
            "R2D2_CoinGame_9f3d3_00039_39_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.PiecewiseSchedu_2021-05-15_16-34-25/checkpoint_000500/checkpoint-500",
        ]
    elif env_name == "ABCoinGame":
        prefix = (
            "/home/maxime/dev-maxime/CLR/vm-data/instance-10-cpu-2/"
            "amTFT/2021_05_17/18_08_40"
        )
        # Prefix to use inside VM
        # prefix = (
        #     os.path.expanduser("~/ray_results/amTFT/2021_05_17/18_08_40")
        # )
        util_load_data_list = [
            "R2D2_SSDMixedMotiveCoinGame_03c71_00000_0_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-41/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00001_1_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-44/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00002_2_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-45/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00003_3_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-45/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00004_4_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-45/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00005_5_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-45/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00006_6_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-46/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00007_7_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-46/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00008_8_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-47/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00009_9_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-19_14-03-47/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00010_10_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_14-03-47/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00011_11_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-40-35/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00012_12_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-40-42/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00013_13_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-40-55/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00014_14_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-41-31/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00015_15_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-41-32/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00016_16_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-42-16/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00017_17_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-42-17/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00018_18_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-42-35/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00019_19_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-42-35/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00020_20_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_17-43-23/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00021_21_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-18-04/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00022_22_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-18-50/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00023_23_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-20-23/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00024_24_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-21-00/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00025_25_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-21-09/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00026_26_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-21-33/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00027_27_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-21-43/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00028_28_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-22-16/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00029_29_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-22-22/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00030_30_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-19_21-22-29/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00031_31_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_00-56-32/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00032_32_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_00-57-59/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00033_33_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_01-00-12/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00034_34_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_01-00-13/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00035_35_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_01-00-48/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00036_36_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_01-01-16/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00037_37_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_01-01-53/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00038_38_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_01-02-36/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_03c71_00039_39_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-20_01-02-46/checkpoint_000500/checkpoint-500",
        ]
        ia_load_data_list = [
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00000_0_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-31/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00001_1_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-34/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00002_2_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-34/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00003_3_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-34/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00004_4_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-34/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00005_5_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-34/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00006_6_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-34/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00007_7_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-35/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00008_8_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-35/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00009_9_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.Pi_2021-05-18_08-39-35/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00010_10_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_08-39-36/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00011_11_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-15-54/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00012_12_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-16-12/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00013_13_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-16-20/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00014_14_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-17-16/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00015_15_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-17-16/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00016_16_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-17-36/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00017_17_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-17-37/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00018_18_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-18-16/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00019_19_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-18-23/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00020_20_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_12-18-50/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00021_21_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-56-24/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00022_22_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-56-36/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00023_23_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-56-49/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00024_24_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-57-37/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00025_25_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-58-34/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00026_26_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-58-35/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00027_27_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-58-35/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00028_28_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-59-18/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00029_29_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-59-39/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00030_30_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_15-59-57/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00031_31_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-35-25/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00032_32_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-36-40/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00033_33_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-36-40/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00034_34_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-37-56/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00035_35_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-38-10/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00036_36_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-38-21/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00037_37_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-38-29/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00038_38_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-38-46/checkpoint_000500/checkpoint-500",
            "R2D2_SSDMixedMotiveCoinGame_8fdcf_00039_39_buffer_size=400000,temperature_schedule=<ray.rllib.utils.schedules.piecewise_schedule.P_2021-05-18_19-38-58/checkpoint_000500/checkpoint-500",
        ]
    else:
        raise ValueError()

    return prefix, util_load_data_list, ia_load_data_list


def _filter_utilitarian_results_with_inequity_aversion_outcomes(hparams):
    hp_cp = copy.deepcopy(hparams)
    _, env_config, _ = get_rllib_config(
        hp_cp,
        (postprocessing.WELFARE_INEQUITY_AVERSION, "inequity_aversion"),
    )

    replicas_paths = [
        os.path.split(os.path.split(ckpt_path)[0])[0]
        for ckpt_path in hparams["load_policy_data"]["Util"]
    ]
    print("replicas_paths", len(replicas_paths))
    filtered_replicas_paths = path_util.filter_list_of_replicates_by_results(
        replicas_paths,
        filter_key=f"policy_reward_mean.{env_config['players_ids'][0]}",
        filter_threshold=79.0,
    )
    print("filtered_replicas_paths", len(filtered_replicas_paths))
    filtered_ckpt = [
        ckpt_path
        for ckpt_path in hparams["load_policy_data"]["Util"]
        if any([el in ckpt_path for el in filtered_replicas_paths])
    ]
    print("filtered_ckpt", len(filtered_ckpt))
    filtered_ckpt = filtered_ckpt[
        : len(hparams["load_policy_data"]["Util"]) // 4
    ]
    print("trunc filtered_ckpt", len(filtered_ckpt))
    hparams["load_policy_data"]["Util"] = filtered_ckpt
    return hparams


def config_and_evaluate_cross_play(experiment_analysis_per_welfare, hp):
    config_eval, env_config, stop, hp_eval = _generate_eval_config(hp)
    config_eval, env_config, stop, hp_eval = _modify_config_to_use_meta_amtft(
        config_eval, env_config, stop, hp_eval
    )
    # config_eval["log_level"] = "DEBUG"
    # config_eval["multiagent"]["policies_to_train"] = ["None"]
    hp_eval["n_self_play_per_checkpoint"] = 0

    config_list_per_welfare = generate_all_configs(
        config_eval, experiment_analysis_per_welfare
    )

    return evaluate_self_play_cross_play(
        config_list_per_welfare, config_eval, env_config, stop, hp_eval
    )


def _modify_config_to_use_meta_amtft(config_eval, env_config, stop, hp_eval):
    amTFT_policies = copy.deepcopy(config_eval["multiagent"]["policies"])

    meta_amTFT_policies = {}
    for policy_id, policy_tuple in amTFT_policies.items():
        (_, observation_space, action_scape, amTFT_config) = policy_tuple
        multi_path_cooperator_config = copy.deepcopy(
            hierarchical.DEFAULT_CONFIG
        )
        multi_path_cooperator_config["nested_policies"] = [
            {
                "Policy_class": amTFT.AmTFTRolloutsTorchPolicy,
                "config_update": copy.deepcopy(amTFT_config),
            },
            {
                "Policy_class": amTFT.AmTFTRolloutsTorchPolicy,
                "config_update": copy.deepcopy(amTFT_config),
            },
        ]
        multi_path_cooperator_config["optimizer"] = copy.deepcopy(
            amTFT_config["optimizer"]
        )
        multi_path_cooperator_config[
            multi_path_cooperator.WORKING_MODE_KEY
        ] = (
            multi_path_cooperator.RANDOM_SWITCHING
            if use_random_switching
            else multi_path_cooperator.PREFERENCE_ORDERING
        )
        multi_path_cooperator_config[
            multi_path_cooperator.PREFERRED_W_PROBA_KEY
        ] = hp_eval[multi_path_cooperator.PREFERRED_W_PROBA_KEY]

        meta_policy_tuple = [
            multi_path_cooperator.MultiPathCooperator,
            observation_space,
            action_scape,
            multi_path_cooperator_config,
        ]
        meta_amTFT_policies[policy_id] = meta_policy_tuple

    config_eval["multiagent"]["policies"] = meta_amTFT_policies

    return config_eval, env_config, stop, hp_eval


def generate_all_configs(config_eval, experiment_analysis_per_welfare):
    config_list_per_welfare = {}
    for (
        group_name_1,
        experiement_analysis_1,
    ) in experiment_analysis_per_welfare.items():
        # if not use_random_switching:
        #     if group_name_1 == "Util":
        #         continue
        checkpoints_in_one_group_1 = (
            utils.restore.extract_checkpoints_from_experiment_analysis(
                experiement_analysis_1
            )
        )
        for (
            group_name_2,
            experiement_analysis_2,
        ) in experiment_analysis_per_welfare.items():
            if not use_random_switching:
                # if group_name_1 == group_name_2:
                #     continue
                if group_name_1 != group_name_2:
                    continue
                # if group_name_2 == "Util":
                #     continue
            checkpoints_in_one_group_2 = (
                utils.restore.extract_checkpoints_from_experiment_analysis(
                    experiement_analysis_2
                )
            )
            assert len(checkpoints_in_one_group_1) == len(
                checkpoints_in_one_group_2
            ), f"{len(checkpoints_in_one_group_1)} == {len(checkpoints_in_one_group_2)}"
            if group_name_1 == group_name_2:
                group_name = "{" + group_name_1 + "}"
            else:
                if use_random_switching:
                    group_name_list = [group_name_1, group_name_2]
                    group_name_list.sort()
                    group_names = ", ".join(group_name_list)
                    group_name = "{" + group_names + "}"
                else:
                    group_name = "{" + group_name_1 + ", " + group_name_2 + "}"

            all_config_wt_ckpt = []
            for ckpt_i in range(len(checkpoints_in_one_group_1)):
                one_config = copy.deepcopy(config_eval)
                for policy_id in one_config["multiagent"]["policies"].keys():
                    policy_config = one_config["multiagent"]["policies"][
                        policy_id
                    ][3]
                    if group_name_1 == group_name_2:
                        policy_config["nested_policies"] = [
                            policy_config["nested_policies"][0]
                        ]

                    ckpt_path_1 = checkpoints_in_one_group_1[ckpt_i]
                    nested_1_config = policy_config["nested_policies"][0][
                        "config_update"
                    ]
                    nested_1_config[restore.LOAD_FROM_CONFIG_KEY] = (
                        ckpt_path_1,
                        policy_id,
                    )
                    nested_1_config["policy_id"] = policy_id
                    if use_min_punish_steps:
                        nested_1_config["min_punish_steps"] = min_punish_steps

                    if group_name_1 != group_name_2:
                        ckpt_path_2 = checkpoints_in_one_group_2[ckpt_i]
                        nested_2_config = policy_config["nested_policies"][1][
                            "config_update"
                        ]
                        nested_2_config[restore.LOAD_FROM_CONFIG_KEY] = (
                            ckpt_path_2,
                            policy_id,
                        )
                        nested_2_config["policy_id"] = policy_id
                        if use_min_punish_steps:
                            nested_2_config[
                                "min_punish_steps"
                            ] = min_punish_steps

                all_config_wt_ckpt.append(one_config)
            config_list_per_welfare[group_name] = all_config_wt_ckpt
        # break
    return config_list_per_welfare


def evaluate_self_play_cross_play(
    config_list_per_welfare, config_eval, env_config, stop, hp_eval
):
    exp_name = os.path.join(hp_eval["exp_name"], "eval")
    evaluator = cross_play.evaluator.SelfAndCrossPlayEvaluator(
        exp_name=exp_name,
        local_mode=hp_eval["debug"],
    )
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=config_eval,
        stop_config=stop,
        policies_to_load_from_checkpoint=copy.deepcopy(
            env_config["players_ids"]
        ),
        config_list_per_welfare=config_list_per_welfare,
        rllib_trainer_class=dqn.r2d2.R2D2Trainer
        if hp_eval["use_r2d2"]
        else dqn.DQNTrainer,
        n_self_play_per_checkpoint=hp_eval["n_self_play_per_checkpoint"],
        n_cross_play_per_checkpoint=hp_eval["n_cross_play_per_checkpoint"],
        to_load_path=hp_eval["load_plot_data"],
    )

    return (
        plot_evaluation(
            hp_eval, evaluator, analysis_metrics_per_mode, env_config
        ),
        evaluator,
    )


if __name__ == "__main__":
    use_r2d2 = True
    debug_mode = False
    main(debug_mode, use_r2d2=use_r2d2)
