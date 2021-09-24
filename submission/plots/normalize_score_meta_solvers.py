import numpy as np

from submission.plots.average_saved_meta_game_results import (
    _get_inputs,
    _preprocess_inputs,
    _raw_data_for_file,
)
from submission.plots.plot_bar_chart_from_saved_results import (
    UTILITARIAN_W,
    EGALITARIAN_W,
    NASH_W,
)

env_name = "IteratedAsymBoS"
welfare_functions = [UTILITARIAN_W, EGALITARIAN_W, NASH_W]
welfare_maximums = [5.0, 2.0, 4.5]
welfare_disagreements_payoffs = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]


def main(debug):
    prefix, files_data, n_players = _get_inputs()
    files_to_process = _preprocess_inputs(prefix, files_data)

    results = {}
    for file, file_data in zip(files_to_process, files_data):
        (
            values_per_replicat_per_player,
            coordination_success,
        ) = _raw_data_for_file(file, n_players, file_data)
        print("file_data", file_data)
        solver = file_data[0]
        normalized_score = _compute_metric_4(
            values_per_replicat_per_player,
            welfare_functions,
            welfare_disagreements_payoffs,
            welfare_maximums,
        )
        print(solver, "normalized_score", normalized_score)
        results[solver] = normalized_score
    print(results)


def compute_normalized_score_meta_solvers(payoffs_per_solvers):
    results = {}
    for solver, mean_payoffs in payoffs_per_solvers.items():
        normalized_score = _compute_metric_4(mean_payoffs)
        print(solver, "normalized_score", normalized_score)
        results[solver] = normalized_score
    print(results)


def _compute_metric_4(
    payoffs_per_solvers,
    welfare_functions_,
    welfare_disagreements_payoffs_,
    welfare_maximums_,
    search_max=False,
    payoffs_for_all_solvers=None,
):
    """
    Metric 4 is:
     max_w { [ w(outcome) - w(disagreement) ]
     / [ max_\pi w(\pi) - w(disagreement)] }
    """

    metrics_for_all_welfares = []
    for welfare_fn, disagreement_payoffs, welfare_max in zip(
        welfare_functions_,
        welfare_disagreements_payoffs_,
        welfare_maximums_,
    ):
        welfare = welfare_fn(np.array(payoffs_per_solvers))
        print("disagreement_payoffs", disagreement_payoffs)
        welfare_disagreement = welfare_fn(np.array([disagreement_payoffs]))
        if search_max:
            welfares_all_solvers = welfare_fn(
                np.array(payoffs_for_all_solvers)
            )
            welfare_max = welfares_all_solvers.max()
        metric_for_one_welfare = (welfare - welfare_disagreement) / (
            welfare_max - welfare_disagreement
        )

        metrics_for_all_welfares.append(metric_for_one_welfare)
    metrics_for_all_welfares = np.stack(metrics_for_all_welfares, axis=1)
    normalized_score = metrics_for_all_welfares.max(axis=1)
    print(
        "payoffs_per_solvers",
        payoffs_per_solvers[normalized_score > 1.0, :],
        normalized_score[normalized_score > 1.0],
    )
    assert np.all(
        metrics_for_all_welfares <= 1.0
    ), metrics_for_all_welfares.max()
    # assert np.all(
    #     metrics_for_all_welfares >= 0.0
    # ), metrics_for_all_welfares.min()
    normalized_score = normalized_score.mean(axis=0)
    return normalized_score


if __name__ == "__main__":
    debug = False
    main(debug)
