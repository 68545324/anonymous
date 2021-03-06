import copy
import json
import logging
import os
import pickle
import random
from typing import Dict

import ray
from ray import tune
from ray.rllib.agents.pg import PGTrainer
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.logger import SafeFallbackEncoder

from submission import utils
from submission.utils import restore, log, miscellaneous
from submission.utils.cross_play import ploter

logger = logging.getLogger(__name__)

RESULTS_SUMMARY_FILENAME_PREFIX = "self_and_cross_play"


class SelfAndCrossPlayEvaluator:
    """
    Utility to run self-play and cross-play performance evaluation.

    Does support only the RLLib API.
    Thus if you are working with Tune, then you will need to use the
    utils.policy.get_tune_policy_class helper to convert your Tune trainer into
    frozen RLLib policies.
    """

    SELF_PLAY_MODE = "self-play"
    CROSS_PLAY_MODE = "cross-play"
    MODES = [CROSS_PLAY_MODE, SELF_PLAY_MODE]

    def __init__(
        self,
        exp_name: str,
        local_mode: bool = False,
        use_random_policy_from_own_checkpoint: bool = False,
        use_wandb: bool = False,
    ):
        """
        You should take a look at examples using this class.
        Any training is deactivated here. Only the worker rollout will evaluate
        your policy on the environment.
        Any exploration is deactivated.

        Works for a unique pair of RLLib policies.

        :param exp_name: Normal exp_name argument provided to tune.run().
        :param use_random_policy_from_own_checkpoint: (optional, default to False)
        """
        self.default_selected_order = 0
        self.running_in_local_mode = local_mode
        self.use_wandb = use_wandb
        self.exp_name, self.exp_parent_dir = log.log_in_current_day_dir(
            exp_name
        )
        self.results_file_name = "SelfAndCrossPlay_save.p"
        self.save_path = os.path.join(
            self.exp_parent_dir, self.results_file_name
        )
        # TODO this var name is not clear enough
        self.use_random_policy_from_own_checkpoint = (
            use_random_policy_from_own_checkpoint
        )

        self.experiment_defined = False
        self._preloading_done = False
        self._provided_data = []

    def perform_evaluation_or_load_data(
        self,
        evaluation_config,
        stop_config,
        policies_to_load_from_checkpoint,
        experiment_analysis_per_welfare: dict = None,
        config_list_per_welfare: dict = None,
        rllib_trainer_class=PGTrainer,
        tune_trainer_class=None,
        n_cross_play_per_checkpoint: int = 1,
        n_self_play_per_checkpoint: int = 1,
        to_load_path: str = None,
    ):
        """

        :param evaluation_config: Normal config argument provided to tune.run().
            This RLLib config will be used to run many similar runs.
            This config will be automatically updated to load the policies from
            the checkpoints you are going to provide.
        :param stop_config: Normal stop_config argument provided to tune.run().
        :param policies_to_load_from_checkpoint:
        :param experiment_analysis_per_welfare: List of the tune_analysis you want to
            extract the groups of checkpoints from. All the checkpoints in these
            tune_analysis will be extracted.
        :param config_list_per_welfare: List of the config you want to
            mix together to produce self and cross play.
        :param rllib_trainer_class: (default is the PGTrainer class) Normal 1st argument (run_or_experiment) provided to
            tune.run(). You should use the one which provides the data flow you need. (Probably a simple PGTrainer will do).
        :param tune_trainer_class: Will only be needed when you are going to evaluate policies created from a Tune
            trainer. You need to provide the class of this trainer.
        :param n_cross_play_per_checkpoint: (int) How many cross-play experiment per checkpoint you want to run.
            They are run randomly against the other checkpoints.
        :param n_self_play_per_checkpoint: (int) How many self-play experiment per checkpoint you want to run.
            More than 1 mean that you are going to run several times the same experiments.
        :param to_load_path: where to load the data from
        :return: data formatted in a way ready for plotting by the plot_results method.
        """
        if to_load_path is None:

            self._use_auto_checkpoint_config = (
                experiment_analysis_per_welfare is not None
            )
            _use_mixing_config = config_list_per_welfare is not None
            msg = (
                "Use either automatic checkpoints retrieval and configuration "
                "(using experiment_analysis_per_welfare) "
                "or use a list of predefined configurations to mix "
                "(using config_list_per_welfare)"
            )
            assert _use_mixing_config is (
                not self._use_auto_checkpoint_config
            ), msg

            self.define_the_experiment_to_run(
                TrainerClass=rllib_trainer_class,
                TuneTrainerClass=tune_trainer_class,
                evaluation_config=evaluation_config,
                stop_config=stop_config,
                policies_to_load_from_checkpoint=policies_to_load_from_checkpoint,
            )
            if self._use_auto_checkpoint_config:
                self.preload_checkpoints_from_tune_results(
                    tune_results=experiment_analysis_per_welfare
                )
            else:
                self._preload_given_configs(config_list_per_welfare)

            analysis_metrics_per_mode = self.evaluate_performances(
                n_self_play_per_checkpoint=n_self_play_per_checkpoint,
                n_cross_play_per_checkpoint=n_cross_play_per_checkpoint,
            )
        else:
            analysis_metrics_per_mode = self.load_results(
                to_load_path=to_load_path
            )
        return analysis_metrics_per_mode

    def _preload_given_configs(self, config_list_per_welfare: Dict[str, list]):
        assert len(self._provided_data) == 0
        self._config_list_per_welfare = config_list_per_welfare
        for (
            group_name,
            one_group_of_configs,
        ) in config_list_per_welfare.items():
            self._provided_data.extend(
                [
                    {"group_name": group_name, "rllib_config": rllib_config}
                    for rllib_config in one_group_of_configs
                ]
            )

        self._n_checkpoints_or_configs = 0
        for groupe_name, config_list in config_list_per_welfare.items():
            self._n_checkpoints_or_configs += len(config_list)
        assert self._n_checkpoints_or_configs == len(self._provided_data)
        msg = f"Found {self._n_checkpoints_or_configs} checkpoints"
        print(msg)
        logger.info(msg)

        self._preloading_done = True

    def define_the_experiment_to_run(
        self,
        evaluation_config: dict,
        stop_config: dict,
        TuneTrainerClass=None,
        TrainerClass=PGTrainer,
        policies_to_load_from_checkpoint: list = ("All"),
    ):
        """
        :param evaluation_config: Normal config argument provided to tune.run().
            This RLLib config will be used to run many similar runs.
            This config will be automatically updated to load the policies
            from the checkpoints you are going to provide.
        :param stop_config: Normal stop_config argument provided to tune.run().
        :param TuneTrainerClass: Will only be needed when you are going to
            evaluate policies created from a Tune
            trainer. You need to provide the class of this trainer.
        :param TrainerClass: (default is the PGTrainer class) The usual 1st
            argument provided to tune.run(). You should use the one which
            provides the data flow you need. (Probably a simple PGTrainer will do).
        :param policies_to_load_from_checkpoint:
        """

        self.TrainerClass = TrainerClass
        self.TuneTrainerClass = TuneTrainerClass
        self.evaluation_config = miscellaneous.set_config_for_evaluation(
            evaluation_config
        )
        self.stop_config = stop_config

        self._config_is_for_two_players()

        self.policies_ids_sorted = sorted(
            list(self.evaluation_config["multiagent"]["policies"].keys())
        )
        self.policies_to_load_from_checkpoint = tuple(
            sorted(
                [
                    policy_id
                    for policy_id in self.policies_ids_sorted
                    if self._is_policy_to_load(
                        policy_id, policies_to_load_from_checkpoint
                    )
                ]
            )
        )

        self.experiment_defined = True

    def _config_is_for_two_players(self):
        assert "multiagent" in self.evaluation_config.keys()
        assert (
            len(self.evaluation_config["multiagent"]["policies"].keys()) == 2
        )

    def _is_policy_to_load(self, policy_id, policies_to_load_from_checkpoint):
        return (
            policy_id in policies_to_load_from_checkpoint
            or "All" in policies_to_load_from_checkpoint
        )

    def preload_checkpoints_from_tune_results(
        self, tune_results: Dict[str, ExperimentAnalysis]
    ):
        """
        :param tune_results: Dict of the tune_analysis you want to extract
            the groups of checkpoints from.
            All the checkpoints in these tune_analysis will be extracted.
        """
        self._extract_groups_of_checkpoints(tune_results)
        self._n_checkpoints_or_configs = len(self._provided_data)
        msg = (
            f"Found {self._n_checkpoints_or_configs} checkpoints in "
            f"{len(tune_results)} tune_results"
        )
        print(msg)
        logger.info(msg)
        self._preloading_done = True

    def _extract_groups_of_checkpoints(
        self, tune_results: Dict[str, ExperimentAnalysis]
    ):
        assert len(self._provided_data) == 0
        for group_name, one_tune_result in tune_results.items():
            self._extract_one_group_of_checkpoints(one_tune_result, group_name)

    def _extract_one_group_of_checkpoints(
        self, one_tune_result: ExperimentAnalysis, group_name
    ):
        checkpoints_in_one_group = (
            utils.restore.extract_checkpoints_from_experiment_analysis(
                one_tune_result
            )
        )
        self._provided_data.extend(
            [
                {"group_name": group_name, "path": checkpoint}
                for checkpoint in checkpoints_in_one_group
            ]
        )

    def evaluate_performances(
        self, n_self_play_per_checkpoint: int, n_cross_play_per_checkpoint: int
    ):
        """
        :param n_self_play_per_checkpoint: (int) How many self-play experiment per checkpoint you want to run.
            More than 1 mean that you are going to run several times the same experiments.
        :param n_cross_play_per_checkpoint: (int) How many cross-play experiment per checkpoint you want to run.
            They are run randomly against the other checkpoints.
        :return: data formatted in a way ready for plotting by the plot_results method.
        """
        assert self._preloading_done, (
            "You must define the checkpoints to evaluate with the "
            "preload_checkpoints_from_tune_results method"
        )
        assert self.experiment_defined, (
            "You must define the evaluation experiment with the "
            "define_the_experiment_to_run method."
        )

        self._validate_number_of_requested_evaluations(
            n_self_play_per_checkpoint, n_cross_play_per_checkpoint
        )
        all_metadata = self._evaluate_performances_in_parallel(
            n_self_play_per_checkpoint, n_cross_play_per_checkpoint
        )
        analysis_metrics_per_mode = self._group_results_and_extract_metrics(
            all_metadata
        )
        self.save_results(analysis_metrics_per_mode)
        return analysis_metrics_per_mode

    def _validate_number_of_requested_evaluations(
        self, n_self_play_per_checkpoint, n_cross_play_per_checkpoint
    ):
        assert n_self_play_per_checkpoint + n_cross_play_per_checkpoint >= 1
        assert n_self_play_per_checkpoint >= 0
        assert n_cross_play_per_checkpoint >= 0
        if n_cross_play_per_checkpoint > 0:
            assert (
                n_cross_play_per_checkpoint
                <= self._n_checkpoints_or_configs - 1
            ), (
                f"n_cross_play_per_checkpoint {n_cross_play_per_checkpoint} "
                f"and self._n_checkpoints_or_configs {self._n_checkpoints_or_configs}"
            )

    def _evaluate_performances_in_parallel(
        self, n_self_play_per_checkpoint, n_cross_play_per_checkpoint
    ):
        master_config, all_metadata = self._prepare_one_master_config_dict(
            n_self_play_per_checkpoint, n_cross_play_per_checkpoint
        )

        results = ray.tune.run(
            self.TrainerClass,
            config=master_config,
            stop=self.stop_config,
            name=os.path.join(self.exp_name, "self_and_cross_play_eval"),
            log_to_file=not self.running_in_local_mode,
            # loggers=None
            # if not self.use_wandb
            # else DEFAULT_LOGGERS + (WandbLogger,),
            verbose=0,
        )

        all_metadata_wt_results = self._add_results_into_metadata(
            all_metadata, results
        )
        return all_metadata_wt_results

    def _prepare_one_master_config_dict(
        self, n_self_play_per_ckpt, n_cross_play_per_ckpt
    ):
        opponents_per_ckpt = self._get_opponents_per_checkpoints(
            n_cross_play_per_ckpt
        )
        if self._use_auto_checkpoint_config:
            (
                all_config_variations,
                all_metadata,
            ) = self._produce_config_variations_from_ckpt(
                n_self_play_per_ckpt, n_cross_play_per_ckpt, opponents_per_ckpt
            )
        else:
            (
                all_config_variations,
                all_metadata,
            ) = self._produce_config_variations_from_configs(
                n_self_play_per_ckpt, n_cross_play_per_ckpt, opponents_per_ckpt
            )

        master_config = self._assemble_in_one_master_config(
            all_config_variations
        )

        return master_config, all_metadata

    def _get_opponents_per_checkpoints(self, n_cross_play_per_checkpoint):
        opponents_per_checkpoint = [
            self._select_opponent_randomly(
                checkpoint_i, n_cross_play_per_checkpoint
            )
            for checkpoint_i in range(self._n_checkpoints_or_configs)
        ]
        return opponents_per_checkpoint

    def _produce_config_variations_from_configs(
        self, n_self_play_per_ckpt, n_cross_play_per_ckpt, opponents_per_ckpt
    ):

        self_plays = [
            self._get_config_for_one_self_play_from_configs(checkpoint_i)
            for checkpoint_i in range(self._n_checkpoints_or_configs)
            for _ in range(n_self_play_per_ckpt)
        ]
        cross_plays = [
            self._get_config_for_one_cross_play_from_configs(
                checkpoint_i, opponents_per_ckpt[checkpoint_i][cross_play_n]
            )
            for checkpoint_i in range(self._n_checkpoints_or_configs)
            for cross_play_n in range(n_cross_play_per_ckpt)
        ]
        msg = (
            f"Prepared {len(self_plays)} self_plays and "
            f"{len(cross_plays)} cross_plays"
        )
        logger.info(msg)
        print(msg)
        all_plays = self_plays + cross_plays

        all_metadata = [play[0] for play in all_plays]
        all_config_variations = [play[1] for play in all_plays]

        return all_config_variations, all_metadata

    def _produce_config_variations_from_ckpt(
        self, n_self_play_per_ckpt, n_cross_play_per_ckpt, opponents_per_ckpt
    ):
        self_plays = [
            self._get_config_for_one_self_play(checkpoint_i)
            for checkpoint_i in range(self._n_checkpoints_or_configs)
            for _ in range(n_self_play_per_ckpt)
        ]
        cross_plays = [
            self._get_config_for_one_cross_play(
                checkpoint_i, opponents_per_ckpt[checkpoint_i][cross_play_n]
            )
            for checkpoint_i in range(self._n_checkpoints_or_configs)
            for cross_play_n in range(n_cross_play_per_ckpt)
        ]
        msg = (
            f"Prepared {len(self_plays)} self_plays and "
            f"{len(cross_plays)} cross_plays"
        )
        logger.info(msg)
        print(msg)
        all_plays = self_plays + cross_plays

        all_metadata = [play[0] for play in all_plays]
        all_config_variations = [play[1] for play in all_plays]

        return all_config_variations, all_metadata

    def _assemble_in_one_master_config(self, all_config_variations):
        master_config = all_config_variations[0]
        all_multiagent_policies = [
            play["multiagent"]["policies"] for play in all_config_variations
        ]
        master_config["multiagent"]["policies"] = tune.grid_search(
            all_multiagent_policies
        )
        return master_config

    def _add_results_into_metadata(self, all_metadata, results):
        for i in range(len(all_metadata)):
            all_metadata[i]["results"] = results.trials[i]
        return all_metadata

    def save_results(self, all_results):
        pickle.dump(all_results, open(self.save_path, "wb"))
        self._save_results_as_json(all_results)

    def _save_results_as_json(self, available_metrics_list):
        metrics = copy.deepcopy(available_metrics_list)
        save_path = self.save_path.split(".")[0:-1] + ["json"]
        save_path = ".".join(save_path)
        with open(save_path, "w") as outfile:
            json.dump(metrics, outfile, cls=SafeFallbackEncoder)

    def load_results(self, to_load_path):
        assert to_load_path.endswith(self.results_file_name), (
            f"to_load_path {to_load_path} should end with "
            f"self.results_file_name {self.results_file_name}"
        )
        all_results = pickle.load(open(to_load_path, "rb"))
        tail, head = os.path.split(to_load_path)
        self.exp_parent_dir = tail
        return all_results

    def _get_config_for_one_self_play_from_configs(self, checkpoint_i):
        metadata = {"mode": self.SELF_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        # Add the checkpoints to load from in the policy_config
        for policy_id in self.policies_to_load_from_checkpoint:
            metadata[policy_id] = {
                "checkpoint_i": checkpoint_i,
            }

            config_copy["multiagent"]["policies"][policy_id][
                3
            ] = self._provided_data[checkpoint_i]["rllib_config"][
                "multiagent"
            ][
                "policies"
            ][
                policy_id
            ][
                3
            ]
        return metadata, config_copy

    def _get_config_for_one_cross_play_from_configs(
        self, own_checkpoint_i, opponent_i
    ):
        metadata = {"mode": self.CROSS_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        if self.use_random_policy_from_own_checkpoint:
            own_position = random.randint(
                0, len(config_copy["multiagent"]["policies"]) - 1
            )
        else:
            own_position = self.default_selected_order

        # Add the checkpoints to load from in the policy_config
        for policy_id in self.policies_to_load_from_checkpoint:
            policy_idx = self.policies_ids_sorted.index(policy_id)
            if own_position == policy_idx:
                checkpoint_idx = own_checkpoint_i
            else:
                checkpoint_idx = opponent_i
            metadata[policy_id] = {
                # "checkpoint_path": checkpoint_path,
                "checkpoint_i": checkpoint_idx,
            }
            config_copy["multiagent"]["policies"][policy_id][
                3
            ] = self._provided_data[checkpoint_idx]["rllib_config"][
                "multiagent"
            ][
                "policies"
            ][
                policy_id
            ][
                3
            ]
        return metadata, config_copy

    def _get_config_for_one_self_play(self, checkpoint_i):
        metadata = {"mode": self.SELF_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        # Add the checkpoints to load from in the policy_config
        for policy_id in self.policies_to_load_from_checkpoint:
            metadata[policy_id] = {
                "checkpoint_path": self._provided_data[checkpoint_i]["path"],
                "checkpoint_i": checkpoint_i,
            }
            policy_config = config_copy["multiagent"]["policies"][policy_id][3]
            policy_config[restore.LOAD_FROM_CONFIG_KEY] = (
                self._provided_data[checkpoint_i]["path"],
                policy_id,
            )
            policy_config["policy_id"] = policy_id
            policy_config["TuneTrainerClass"] = self.TuneTrainerClass
        return metadata, config_copy

    def _get_config_for_one_cross_play(self, own_checkpoint_i, opponent_i):
        metadata = {"mode": self.CROSS_PLAY_MODE}
        config_copy = copy.deepcopy(self.evaluation_config)

        if self.use_random_policy_from_own_checkpoint:
            own_position = random.randint(
                0, len(config_copy["multiagent"]["policies"]) - 1
            )
        else:
            own_position = self.default_selected_order

        # Add the checkpoints to load from in the policy_config
        for policy_id in self.policies_to_load_from_checkpoint:
            policy_idx = self.policies_ids_sorted.index(policy_id)
            if own_position == policy_idx:
                checkpoint_idx = own_checkpoint_i
            else:
                checkpoint_idx = opponent_i
            checkpoint_path = self._provided_data[checkpoint_idx]["path"]
            metadata[policy_id] = {
                "checkpoint_path": checkpoint_path,
                "checkpoint_i": checkpoint_idx,
            }
            policy_config = config_copy["multiagent"]["policies"][policy_id][3]
            policy_config[restore.LOAD_FROM_CONFIG_KEY] = (
                checkpoint_path,
                policy_id,
            )
            policy_config["policy_id"] = policy_id
            policy_config["TuneTrainerClass"] = self.TuneTrainerClass
        return metadata, config_copy

    def _select_opponent_randomly(
        self, checkpoint_i, n_cross_play_per_checkpoint
    ):
        checkpoint_list_minus_i = list(range(len(self._provided_data)))
        checkpoint_list_minus_i.pop(checkpoint_i)
        opponents = random.sample(
            checkpoint_list_minus_i, n_cross_play_per_checkpoint
        )
        return opponents

    def _split_metadata_per_mode(self, all_results):
        return {
            mode: [report for report in all_results if report["mode"] == mode]
            for mode in self.MODES
        }

    def _split_metadata_per_group_pair_id(self, metadata_for_one_mode, mode):
        analysis_per_group_pair_id = []

        experiment_analysis = [
            metadata["results"] for metadata in metadata_for_one_mode
        ]
        pairs_of_group_names = [
            self._get_pair_of_group_names(metadata)
            for metadata in metadata_for_one_mode
        ]
        ids_of_pairs_of_groups = [
            self._get_id_of_pair_of_group_names(one_pair_of_group_names)
            for one_pair_of_group_names in pairs_of_group_names
        ]
        group_pair_ids_in_this_mode = sorted(set(ids_of_pairs_of_groups))

        for group_pair_id in list(group_pair_ids_in_this_mode):
            (
                filtered_analysis_list,
                one_pair_of_group_names,
            ) = self._find_and_group_results_for_one_group_pair_id(
                group_pair_id,
                experiment_analysis,
                ids_of_pairs_of_groups,
                pairs_of_group_names,
            )
            analysis_per_group_pair_id.append(
                (
                    mode,
                    filtered_analysis_list,
                    group_pair_id,
                    one_pair_of_group_names,
                )
            )
        return analysis_per_group_pair_id

    def _get_pair_of_group_names(self, metadata):
        checkpoints_idx_used = {
            policy_id: metadata[policy_id]["checkpoint_i"]
            for policy_id in self.policies_to_load_from_checkpoint
        }
        # TODO could be simplified (metadata could contain this info and not
        #  the checkpoint_i?)
        if self._use_auto_checkpoint_config:
            pair_of_group_names = {
                policy_id: self._provided_data[checkpoint_i]["group_name"]
                for policy_id, checkpoint_i in checkpoints_idx_used.items()
            }
        else:
            pair_of_group_names = {
                policy_id: self._provided_data[checkpoint_i]["group_name"]
                for policy_id, checkpoint_i in checkpoints_idx_used.items()
            }
        return pair_of_group_names

    def _get_id_of_pair_of_group_names(self, pair_of_group_names):
        ordered_pair_of_group_names = [
            pair_of_group_names[policy_id]
            for policy_id in self.policies_to_load_from_checkpoint
        ]
        id_of_pair_of_group_names = "".join(ordered_pair_of_group_names)
        return id_of_pair_of_group_names

    def _find_and_group_results_for_one_group_pair_id(
        self, group_pair_id, tune_analysis, group_pair_ids, group_pair_names
    ):
        filtered_group_pair_names, filtered_tune_analysis = [], []
        for one_tune_analysis, id_, pair_of_names in zip(
            tune_analysis, group_pair_ids, group_pair_names
        ):
            if id_ == group_pair_id:
                filtered_tune_analysis.append(one_tune_analysis)
                filtered_group_pair_names.append(pair_of_names)

        filtered_ids = [
            self._get_id_of_pair_of_group_names(one_pair_of_names)
            for one_pair_of_names in filtered_group_pair_names
        ]
        assert len(set(filtered_ids)) == 1
        one_pair_of_names = filtered_group_pair_names[0]

        return filtered_tune_analysis, one_pair_of_names

    def _group_results_and_extract_metrics(self, all_metadata_wt_results):
        # TODO improve the design to remove these unclear names
        analysis_per_mode_per_group_pair_id = (
            self._split_results_per_mode_and_group_pair_id(
                all_metadata_wt_results
            )
        )
        analysis_metrics_per_mode_per_group_pair_id = (
            self._extract_all_metrics(analysis_per_mode_per_group_pair_id)
        )
        return analysis_metrics_per_mode_per_group_pair_id

    def _split_results_per_mode_and_group_pair_id(
        self, all_metadata_wt_results
    ):
        analysis_per_mode = []

        metadata_per_modes = self._split_metadata_per_mode(
            all_metadata_wt_results
        )
        for mode, metadata_for_one_mode in metadata_per_modes.items():
            analysis_per_mode.extend(
                self._split_metadata_per_group_pair_id(
                    metadata_for_one_mode, mode
                )
            )

        return analysis_per_mode

    def _extract_all_metrics(self, analysis_per_mode):
        analysis_metrics_per_mode = []
        for mode_i, mode_data in enumerate(analysis_per_mode):
            mode, analysis_list, group_pair_id, group_pair_name = mode_data

            available_metrics_list = []
            for trial in analysis_list:
                available_metrics = trial.metric_analysis
                available_metrics_list.append(available_metrics)
            analysis_metrics_per_mode.append(
                (mode, available_metrics_list, group_pair_id, group_pair_name)
            )
        return analysis_metrics_per_mode

    def plot_results(
        self,
        analysis_metrics_per_mode,
        plot_config,
        x_axis_metric,
        y_axis_metric,
    ):

        vanilla_plot_path = self._plot_as_provided(
            analysis_metrics_per_mode,
            plot_config,
            x_axis_metric,
            y_axis_metric,
        )

        if plot_config.plot_max_n_points is not None:
            plot_config.plot_max_n_points *= 2

        self._plot_merge_self_cross(
            analysis_metrics_per_mode,
            plot_config,
            x_axis_metric,
            y_axis_metric,
        )
        self._plot_merge_same_and_diff_pref(
            analysis_metrics_per_mode,
            plot_config,
            x_axis_metric,
            y_axis_metric,
        )
        return vanilla_plot_path

    def _plot_as_provided(
        self,
        analysis_metrics_per_mode,
        plot_config,
        x_axis_metric,
        y_axis_metric,
    ):
        vanilla_plot_path = self._plot_one_time(
            analysis_metrics_per_mode,
            plot_config,
            x_axis_metric,
            y_axis_metric,
        )
        return vanilla_plot_path

    def _plot_merge_self_cross(
        self,
        analysis_metrics_per_mode,
        plot_config,
        x_axis_metric,
        y_axis_metric,
    ):
        self._plot_one_time_with_prefix_and_preprocess(
            "_self_cross",
            "_merge_into_self_and_cross",
            analysis_metrics_per_mode,
            plot_config,
            x_axis_metric,
            y_axis_metric,
            save_plotter=True,
        )

    def _plot_merge_same_and_diff_pref(
        self,
        analysis_metrics_per_mode,
        plot_config,
        x_axis_metric,
        y_axis_metric,
    ):

        self._plot_one_time_with_prefix_and_preprocess(
            "_same_and_diff_pref",
            "_merge_into_cross_same_pref_diff_pref",
            analysis_metrics_per_mode,
            plot_config,
            x_axis_metric,
            y_axis_metric,
        )

    def _plot_one_time_with_prefix_and_preprocess(
        self,
        prefix: str,
        preprocess: str,
        analysis_metrics_per_mode,
        plot_config,
        x_axis_metric,
        y_axis_metric,
        save_plotter=False,
    ):
        initial_filename_prefix = plot_config.filename_prefix
        plot_config.filename_prefix += prefix
        metrics_for_same_pref_diff_pref = getattr(self, preprocess)(
            analysis_metrics_per_mode
        )
        self._plot_one_time(
            metrics_for_same_pref_diff_pref,
            plot_config,
            x_axis_metric,
            y_axis_metric,
            save_plotter=save_plotter,
        )
        plot_config.filename_prefix = initial_filename_prefix

    def _merge_into_self_and_cross(self, analysis_metrics_per_mode):
        metrics_for_self_and_cross = []
        for selected_play_mode in self.MODES:
            metrics_for_self_and_cross.append(
                (selected_play_mode, [], "", None)
            )
            for (
                play_mode,
                metrics,
                pair_name,
                pair_tuple,
            ) in analysis_metrics_per_mode:
                if play_mode == selected_play_mode:
                    metrics_for_self_and_cross[-1][1].extend(metrics)
        return metrics_for_self_and_cross

    def _merge_into_cross_same_pref_diff_pref(self, analysis_metrics_per_mode):
        analysis_metrics_per_mode_wtout_self_play = self._copy_wtout_self_play(
            analysis_metrics_per_mode
        )
        one_pair_of_group_names = analysis_metrics_per_mode[0][3]
        metrics_for_same_pref_diff_pref = [
            (
                self.CROSS_PLAY_MODE,
                [],
                "same_prefsame_pref",
                {k: "same_pref" for k in one_pair_of_group_names.keys()},
            ),
            (
                self.CROSS_PLAY_MODE,
                [],
                "diff_prefdiff_pref",
                {k: "diff_pref" for k in one_pair_of_group_names.keys()},
            ),
        ]
        for (
            play_mode,
            metrics,
            pair_name,
            pair_of_group_names,
        ) in analysis_metrics_per_mode_wtout_self_play:
            groups_names = list(pair_of_group_names.values())
            assert len(groups_names) == 2
            if groups_names[0] == groups_names[1]:
                metrics_for_same_pref_diff_pref[0][1].extend(metrics)
            else:
                metrics_for_same_pref_diff_pref[1][1].extend(metrics)
        if len(metrics_for_same_pref_diff_pref[1][1]) == 0:
            metrics_for_same_pref_diff_pref.pop(1)
        if len(metrics_for_same_pref_diff_pref[0][1]) == 0:
            metrics_for_same_pref_diff_pref.pop(0)
        return metrics_for_same_pref_diff_pref

    def _copy_wtout_self_play(self, analysis_metrics_per_mode):
        analysis_metrics_per_mode_wtout_self_play = []
        for (
            play_mode,
            metrics,
            pair_name,
            pair_tuple,
        ) in analysis_metrics_per_mode:
            if play_mode == self.CROSS_PLAY_MODE:
                analysis_metrics_per_mode_wtout_self_play.append(
                    (self.CROSS_PLAY_MODE, metrics, pair_name, pair_tuple)
                )
        return analysis_metrics_per_mode_wtout_self_play

    def _plot_one_time(
        self,
        analysis_metrics_per_mode,
        plot_config,
        x_axis_metric,
        y_axis_metric,
        save_plotter=False,
    ):
        plotter = ploter.SelfAndCrossPlayPlotter()
        plot_path = plotter.plot_results(
            exp_parent_dir=self.exp_parent_dir,
            metrics_per_mode=analysis_metrics_per_mode,
            plot_config=plot_config,
            x_axis_metric=x_axis_metric,
            y_axis_metric=y_axis_metric,
        )
        if save_plotter:
            self.plotter = plotter
        return plot_path
