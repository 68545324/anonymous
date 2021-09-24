import logging
import random
from typing import TYPE_CHECKING

from ray.rllib.utils.threading import with_lock

if TYPE_CHECKING:
    pass

from submission.algos import hierarchical
from submission.algos.amTFT.base import (
    AmTFTReferenceClass,
    OWN_SELFISH_POLICY_IDX,
)

WORKING_MODE_KEY = "working_mode"
PREFERENCE_ORDERING = "preference_ordering"
RANDOM_SWITCHING = "random switching"

PREFERRED_W_PROBA_KEY = "preferred_w_proba"

logger = logging.getLogger(__name__)


class MultiPathCooperator(
    hierarchical.HierarchicalTorchPolicy, AmTFTReferenceClass
):
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(observation_space, action_space, config, **kwargs)

        self.working_mode = config[WORKING_MODE_KEY]
        self._list_algo_idx = list(range(len(self.algorithms)))
        self.prefered_proba = config.get(
            PREFERRED_W_PROBA_KEY, 1 / len(self.algorithms)
        )
        self._select_random_algo()

    def on_observation_fn(self, *args, **kwargs):
        for algo in self.algorithms:
            algo.on_observation_fn(*args, **kwargs)

    def on_episode_step(
        self,
        policy_id,
        policy,
        policy_ids,
        episode,
        worker,
        base_env,
        env_index,
        *args,
        **kwargs,
    ):

        assert worker.policy_map[policy_id] is self
        active_welfare_doesnt_match = False
        for algo_idx, algo in enumerate(self.algorithms):
            # Needed to make the rollouts work fine inside the amTFT algo
            worker.policy_map[policy_id] = self.algorithms[algo_idx]
            algo.on_episode_step(
                policy_id,
                policy,
                policy_ids,
                episode,
                worker,
                base_env,
                env_index,
                *args,
                **kwargs,
            )

            if self.working_mode == RANDOM_SWITCHING:
                opponent_not_using_this_welfare = (
                    algo.total_debit != 0.0 or algo.n_steps_to_punish != 0
                )
                if opponent_not_using_this_welfare:
                    # Never punish
                    algo.total_debit = 0
                    algo.n_steps_to_punish = 0
                    if algo_idx == self.active_algo_idx:
                        active_welfare_doesnt_match = True

        if self.working_mode == RANDOM_SWITCHING:
            # Randomly switch between algo if the opponent is not using the same
            # welfare fn
            if active_welfare_doesnt_match:
                self._select_random_algo()

        worker.policy_map[policy_id] = self

    def _select_random_algo(self):
        if len(self.algorithms) == 1:
            self.active_algo_idx = 0
        else:
            select_prefered_algo = random.random() < self.prefered_proba
            if select_prefered_algo:
                self.active_algo_idx = 0
            else:
                # Select one of the other algo with equal chance
                while self.active_algo_idx == 0:
                    self.active_algo_idx = random.choice(self._list_algo_idx)

    def on_episode_start(self, *args, **kwargs):
        for algo in self.algorithms:
            algo.on_episode_start(*args, **kwargs)

    def on_episode_end(self, *args, **kwargs):
        for algo in self.algorithms:
            algo.on_episode_end(*args, **kwargs)

    @with_lock
    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):

        if self.working_mode == PREFERENCE_ORDERING:
            return self._compute_action_helper_wt_preference_ordering(
                input_dict, state_batches, seq_lens, explore, timestep
            )
        elif self.working_mode == RANDOM_SWITCHING:
            return self._compute_action_helper_wt_random_switching(
                input_dict, state_batches, seq_lens, explore, timestep
            )

    def _compute_action_helper_wt_preference_ordering(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        all_algo_outputs = []
        all_algo_punishing = []
        for algo_idx, algo in enumerate(self.algorithms):
            outputs = algo._compute_action_helper(
                input_dict, state_batches, seq_lens, explore, timestep
            )

            assert not algo.performing_rollouts
            assert not algo.use_opponent_policies

            punishing = algo.active_algo_idx == OWN_SELFISH_POLICY_IDX

            all_algo_outputs.append(outputs)
            all_algo_punishing.append(punishing)
            self._to_log[f"algo_idx {algo_idx} punishing"] = punishing

        return self._return_first_cooperative_or_first_punish(
            all_algo_outputs, all_algo_punishing
        )

    def _return_first_cooperative_or_first_punish(
        self, all_algo_outputs, all_algo_punishing
    ):
        for idx, (algo_outputs, algo_punishing) in enumerate(
            zip(all_algo_outputs, all_algo_punishing)
        ):
            # First cooperative algo
            if not algo_punishing:
                self._to_log[f"first_cooperative_algo"] = idx
                self._to_log[f"use_punishing_algo"] = False
                return algo_outputs

        # First punishing algo
        self._to_log[f"use_punishing_algo"] = True
        self._to_log[f"first_cooperative_algo"] = False
        return all_algo_outputs[0]

    def _compute_action_helper_wt_random_switching(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        all_algo_outputs = []
        all_algo_punishing = []
        for algo_idx, algo in enumerate(self.algorithms):
            outputs = algo._compute_action_helper(
                input_dict, state_batches, seq_lens, explore, timestep
            )

            assert not algo.performing_rollouts
            assert not algo.use_opponent_policies

            punishing = algo.active_algo_idx == OWN_SELFISH_POLICY_IDX
            assert not punishing

            all_algo_outputs.append(outputs)
            all_algo_punishing.append(punishing)
            self._to_log[f"algo_idx {algo_idx} punishing"] = punishing

        return all_algo_outputs[self.active_algo_idx]
