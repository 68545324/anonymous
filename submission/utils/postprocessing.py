import logging
from typing import List, Dict, TYPE_CHECKING

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.evaluation.postprocessing import discount_cumsum
from ray.rllib.evaluation.sampler import _get_or_raise
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.utils.schedules import Schedule

from submission.utils.miscellaneous import (
    assert_if_key_in_dict_then_args_are_none,
    read_from_dict_default_to_args,
)

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

logger = logging.getLogger(__name__)

WELFARE_UTILITARIAN = "utilitarian_welfare"
WELFARE_INEQUITY_AVERSION = "inequity_aversion_welfare"
WELFARE_NASH = "nash_welfare"
WELFARE_EGALITARIAN = "egalitarian_welfare"

WELFARES = (WELFARE_UTILITARIAN, WELFARE_INEQUITY_AVERSION)

OPPONENT_ACTIONS = "opponent_actions"
OPPONENT_NEGATIVE_REWARD = "opponent_negative_reward"
DISCOUNTED_RETURNS = "discounted_returns"
REWARDS_UNDER_DEFECTION = "rewards_under_defection"

ADD_UTILITARIAN_WELFARE = "add_utilitarian_welfare"
ADD_INEQUITY_AVERSION_WELFARE = "add_inequity_aversion_welfare"
ADD_NASH_WELFARE = "add_nash_welfare"
ADD_EGALITARIAN_WELFARE = "add_egalitarian_welfare"
ADD_OPPONENT_ACTION = "add_opponent_action"
ADD_OPPONENT_NEG_REWARD = "add_opponent_neg_reward"

ADD_WELFARE_CONFIG_KEYS = (
    ADD_UTILITARIAN_WELFARE,
    ADD_INEQUITY_AVERSION_WELFARE,
)


def welfares_postprocessing_fn(
    add_utilitarian_welfare: bool = None,
    add_egalitarian_welfare: bool = None,
    add_nash_welfare: bool = None,
    add_opponent_action: bool = None,
    add_opponent_neg_reward: bool = None,
    add_inequity_aversion_welfare: bool = None,
    inequity_aversion_alpha: float = None,
    inequity_aversion_beta: float = None,
    inequity_aversion_gamma: float = None,
    inequity_aversion_lambda: float = None,
    additional_fn: list = [],
):
    """
    Generate a postprocess_fn that first add a welfare if you chose so and
    then call a list of additional postprocess_fn to further modify the
    samples.

    The parameters used to add a welfare can be given as arguments or
    will be read in the policy config dict (this should be preferred since this
    allows for hyperparameter search over these parameters with Tune).When
    read from the config, they are read from the keys defined by
    ADD_INEQUITY_AVERSION_WELFARE and similars. The value oassociated with
    this key must be a tuple if you provide several parameters.

    :param add_utilitarian_welfare:
    :param add_egalitarian_welfare:
    :param add_nash_welfare:
    :param add_opponent_action:
    :param add_opponent_neg_reward:
    :param add_inequity_aversion_welfare:
    :param inequity_aversion_alpha: coeff of disvalue when own discounted
        reward is lower than opponent
    :param inequity_aversion_beta: coeff of disvalue when own discounted
        reward is higher than opponent
    :param inequity_aversion_gamma: usual discount factor
    :param inequity_aversion_lambda: discount factor specific to
        inequity_aversion
    :param additional_fn: a list of additional postprocess_fn you want to
        execute after adding the welfare
    :return:
    """

    # TODO refactor into instanciating an object (from a new class) and
    #  returning one of it method OR use named tuple
    def postprocess_fn(policy, sample_batch, other_agent_batches, episode):

        if other_agent_batches is None:
            logger.warning(
                "no other_agent_batches given for welfare " "postprocessing"
            )
            return sample_batch
        _assert_using_config_xor_args(policy)
        parameters = _read_parameters_from_config_default_to_args(policy)
        parameters = _get_values_from_any_scheduler(parameters, policy)
        sample_batch = _add_welfare_to_own_batch(
            sample_batch, other_agent_batches, episode, policy, *parameters
        )
        sample_batch = _call_list_of_additional_fn(
            additional_fn, sample_batch, other_agent_batches, episode, policy
        )

        return sample_batch

    def _assert_using_config_xor_args(policy):
        assert_if_key_in_dict_then_args_are_none(
            policy.config, ADD_UTILITARIAN_WELFARE, add_utilitarian_welfare
        )
        assert_if_key_in_dict_then_args_are_none(
            policy.config,
            ADD_INEQUITY_AVERSION_WELFARE,
            add_inequity_aversion_welfare,
            inequity_aversion_alpha,
            inequity_aversion_beta,
            inequity_aversion_gamma,
            inequity_aversion_lambda,
        )
        assert_if_key_in_dict_then_args_are_none(
            policy.config, ADD_NASH_WELFARE, add_nash_welfare
        )
        assert_if_key_in_dict_then_args_are_none(
            policy.config, ADD_EGALITARIAN_WELFARE, add_egalitarian_welfare
        )
        assert_if_key_in_dict_then_args_are_none(
            policy.config, ADD_OPPONENT_ACTION, add_opponent_action
        )
        assert_if_key_in_dict_then_args_are_none(
            policy.config, ADD_OPPONENT_NEG_REWARD, add_opponent_neg_reward
        )

    def _read_parameters_from_config_default_to_args(policy):
        add_utilitarian_w = read_from_dict_default_to_args(
            policy.config, ADD_UTILITARIAN_WELFARE, add_utilitarian_welfare
        )
        (
            add_ia_w,
            ia_alpha,
            ia_beta,
            ia_gamma,
            ia_lambda,
        ) = read_from_dict_default_to_args(
            policy.config,
            ADD_INEQUITY_AVERSION_WELFARE,
            add_inequity_aversion_welfare,
            inequity_aversion_alpha,
            inequity_aversion_beta,
            inequity_aversion_gamma,
            inequity_aversion_lambda,
        )
        add_nash_w = read_from_dict_default_to_args(
            policy.config, ADD_NASH_WELFARE, add_nash_welfare
        )
        add_egalitarian_w = read_from_dict_default_to_args(
            policy.config, ADD_EGALITARIAN_WELFARE, add_egalitarian_welfare
        )
        add_opponent_a = read_from_dict_default_to_args(
            policy.config, ADD_OPPONENT_ACTION, add_opponent_action
        )
        add_opponent_neg_r = read_from_dict_default_to_args(
            policy.config, ADD_OPPONENT_NEG_REWARD, add_opponent_neg_reward
        )

        return (
            add_utilitarian_w,
            add_ia_w,
            ia_alpha,
            ia_beta,
            ia_gamma,
            ia_lambda,
            add_nash_w,
            add_egalitarian_w,
            add_opponent_a,
            add_opponent_neg_r,
        )

    def _get_values_from_any_scheduler(parameters, policy):
        logger.debug("_get_values_from_any_scheduler")
        new_parameters = []
        for param in parameters:
            if isinstance(param, Schedule):
                value_from_scheduler = param.value(policy.global_timestep)
                new_parameters.append(value_from_scheduler)
            else:
                new_parameters.append(param)
        return new_parameters

    def _add_welfare_to_own_batch(
        sample_batch, other_agent_batches, episode, policy, *parameters
    ):

        (
            add_utilitarian_w,
            add_ia_w,
            ia_alpha,
            ia_beta,
            ia_gamma,
            ia_lambda,
            add_nash_w,
            add_egalitarian_w,
            add_opponent_a,
            add_opponent_neg_r,
        ) = parameters

        _assert_working_on_one_full_epi(sample_batch)

        if add_utilitarian_w:
            logger.debug(
                f"add utilitarian welfare to batch of policy" f" {policy}"
            )
            opp_batches = [v[1] for v in other_agent_batches.values()]
            sample_batch = _add_utilitarian_welfare_to_batch(
                sample_batch, opp_batches, policy
            )
        if add_ia_w:
            logger.debug(
                f"add inequity aversion welfare to batch of policy"
                f" {policy}"
            )
            _assert_two_players_env(other_agent_batches)
            opp_batch = _get_opp_batch(other_agent_batches)
            sample_batch = _add_inequity_aversion_welfare_to_batch(
                sample_batch,
                opp_batch,
                alpha=ia_alpha,
                beta=ia_beta,
                gamma=ia_gamma,
                lambda_=ia_lambda,
                policy=policy,
            )
        if add_nash_w:
            _assert_two_players_env(other_agent_batches)
            opp_batch = _get_opp_batch(other_agent_batches)
            sample_batch = _add_nash_welfare_to_batch(
                sample_batch, opp_batch, policy
            )
        if add_egalitarian_w:
            _assert_two_players_env(other_agent_batches)
            opp_batch = _get_opp_batch(other_agent_batches)
            sample_batch = _add_egalitarian_welfare_to_batch(
                sample_batch, opp_batch, policy
            )
        if add_opponent_a:
            _assert_two_players_env(other_agent_batches)
            opp_batch = _get_opp_batch(other_agent_batches)
            sample_batch = _add_opponent_action_to_batch(
                sample_batch, opp_batch, policy
            )
        if add_opponent_neg_r:
            _assert_two_players_env(other_agent_batches)
            opp_batch = _get_opp_batch(other_agent_batches)
            sample_batch = _add_opponent_neg_reward_to_batch(
                sample_batch, opp_batch, policy
            )

        return sample_batch

    return postprocess_fn


def _assert_working_on_one_full_epi(sample_batch):
    assert (
        len(set(sample_batch[sample_batch.EPS_ID])) == 1
    ), "designed to work on one complete episode"
    assert (
        not any(sample_batch[sample_batch.DONES][:-1])
        or sample_batch[sample_batch.DONES][-1]
    ), (
        "welfare postprocessing is designed to work on one complete episode, "
        f"dones: {sample_batch[sample_batch.DONES]}"
    )


def _call_list_of_additional_fn(
    additional_fn, sample_batch, other_agent_batches, episode, policy
):
    for postprocessing_function in additional_fn:
        sample_batch = postprocessing_function(
            sample_batch, other_agent_batches, episode, policy
        )

    return sample_batch


def _assert_two_players_env(other_agent_batches):
    assert len(other_agent_batches) == 1


def _get_opp_batch(other_agent_batches):
    return other_agent_batches[list(other_agent_batches.keys())[0]][1]


def _add_utilitarian_welfare_to_batch(
    sample_batch: SampleBatch, opp_ag_batchs: List[SampleBatch], policy=None
) -> SampleBatch:
    all_batchs_rewards = [sample_batch[sample_batch.REWARDS]] + [
        opp_batch[opp_batch.REWARDS] for opp_batch in opp_ag_batchs
    ]
    sample_batch[WELFARE_UTILITARIAN] = np.array(
        [sum(reward_points) for reward_points in zip(*all_batchs_rewards)]
    )

    _ = _log_in_policy(
        np.sum(sample_batch[WELFARE_UTILITARIAN]),
        f"sum_over_epi_{WELFARE_UTILITARIAN}",
        policy,
    )
    return sample_batch


def _log_in_policy(value, name_value, policy=None):
    if policy is not None:
        if hasattr(policy, "to_log"):
            policy.to_log[name_value] = value
        else:
            setattr(policy, "to_log", {name_value: value})
    return policy


def _add_opponent_action_to_batch(
    sample_batch: SampleBatch, opp_ag_batch: SampleBatch, policy=None
) -> SampleBatch:
    sample_batch[OPPONENT_ACTIONS] = opp_ag_batch[opp_ag_batch.ACTIONS]
    _ = _log_in_policy(
        sample_batch[OPPONENT_ACTIONS][-1],
        f"last_{OPPONENT_ACTIONS}",
        policy,
    )
    return sample_batch


def _add_opponent_neg_reward_to_batch(
    sample_batch: SampleBatch, opp_ag_batch: SampleBatch, policy=None
) -> SampleBatch:
    sample_batch[OPPONENT_NEGATIVE_REWARD] = np.array(
        [-opp_r for opp_r in opp_ag_batch[opp_ag_batch.REWARDS]]
    )
    _ = _log_in_policy(
        np.sum(sample_batch[OPPONENT_NEGATIVE_REWARD]),
        f"sum_over_epi_{OPPONENT_NEGATIVE_REWARD}",
        policy,
    )
    return sample_batch


def _add_inequity_aversion_welfare_to_batch(
    sample_batch: SampleBatch,
    opp_ag_batch: SampleBatch,
    alpha: float,
    beta: float,
    gamma: float,
    lambda_: float,
    policy=None,
) -> SampleBatch:
    """
    :param sample_batch: SampleBatch to mutate
    :param opp_ag_batchs:
    :param alpha: coeff of disvalue when own discounted reward is lower than
        opponent
    :param beta: coeff of disvalue when own discounted reward is higher than
        opponent
    :param gamma: discount factor
    :return: sample_batch mutated with WELFARE_INEQUITY_AVERSION added
    """

    own_rewards = np.array(sample_batch[sample_batch.REWARDS])
    opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    own_rewards = np.flip(own_rewards)
    opp_rewards = np.flip(opp_rewards)
    delta = discount_cumsum(own_rewards, gamma * lambda_) - discount_cumsum(
        opp_rewards, gamma * lambda_
    )
    delta = np.flip(delta)
    disvalue_lower_than_opp = alpha * (-delta)
    disvalue_higher_than_opp = beta * delta
    disvalue_lower_than_opp[disvalue_lower_than_opp < 0] = 0
    disvalue_higher_than_opp[disvalue_higher_than_opp < 0] = 0

    welfare = (
        sample_batch[sample_batch.REWARDS]
        - disvalue_lower_than_opp
        - disvalue_higher_than_opp
    )

    sample_batch[WELFARE_INEQUITY_AVERSION] = welfare

    policy = _log_in_policy(
        np.sum(sample_batch[WELFARE_INEQUITY_AVERSION]),
        f"sum_over_epi_{WELFARE_INEQUITY_AVERSION}",
        policy,
    )
    return sample_batch


def _add_nash_welfare_to_batch(
    sample_batch: SampleBatch, opp_ag_batch: SampleBatch, policy=None
) -> SampleBatch:
    own_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    own_rewards_under_defection = np.array(
        opp_ag_batch[REWARDS_UNDER_DEFECTION]
    )
    opp_rewards_under_defection = np.array(
        opp_ag_batch[REWARDS_UNDER_DEFECTION]
    )

    own_delta = sum(own_rewards) - sum(own_rewards_under_defection)
    opp_delta = sum(opp_rewards) - sum(opp_rewards_under_defection)

    nash_welfare = own_delta * opp_delta

    sample_batch[WELFARE_NASH] = (
        [0.0] * (len(sample_batch[sample_batch.REWARDS]) - 1)
    ) + [nash_welfare]
    policy = _log_in_policy(
        np.sum(sample_batch[WELFARE_NASH]),
        f"sum_over_epi_{WELFARE_NASH}",
        policy,
    )
    return sample_batch


def _add_egalitarian_welfare_to_batch(
    sample_batch: SampleBatch, opp_ag_batch: SampleBatch, policy=None
) -> SampleBatch:
    own_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    opp_rewards = np.array(opp_ag_batch[opp_ag_batch.REWARDS])
    own_rewards_under_defection = np.array(
        opp_ag_batch[REWARDS_UNDER_DEFECTION]
    )
    opp_rewards_under_defection = np.array(
        opp_ag_batch[REWARDS_UNDER_DEFECTION]
    )

    own_delta = sum(own_rewards) - sum(own_rewards_under_defection)
    opp_delta = sum(opp_rewards) - sum(opp_rewards_under_defection)

    egalitarian_welfare = min(own_delta, opp_delta)

    sample_batch[WELFARE_EGALITARIAN] = (
        [0.0] * (len(sample_batch[sample_batch.REWARDS]) - 1)
    ) + [egalitarian_welfare]
    policy = _log_in_policy(
        np.sum(sample_batch[WELFARE_EGALITARIAN]),
        f"sum_over_epi_{WELFARE_EGALITARIAN}",
        policy,
    )
    return sample_batch


class OverwriteRewardWtWelfareCallback(DefaultCallbacks):
    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: MultiAgentEpisode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, SampleBatch],
        **kwargs,
    ):

        assert (
            sum([k in WELFARES for k in postprocessed_batch.keys()]) <= 1
        ), "only one welfare must be available"

        for welfare_key in WELFARES:
            if welfare_key in postprocessed_batch.keys():
                postprocessed_batch[
                    postprocessed_batch.REWARDS
                ] = postprocessed_batch[welfare_key]
                msg = (
                    f"Overwrite the reward of agent_id {agent_id} "
                    f"with the value from the"
                    f" welfare_key {welfare_key}"
                )
                print(msg)
                logger.debug(msg)
                break

        return postprocessed_batch


def apply_preprocessors(worker, raw_observation, policy_id):
    prep_obs = _get_or_raise(worker.preprocessors, policy_id).transform(
        raw_observation
    )
    filtered_obs = _get_or_raise(worker.filters, policy_id)(prep_obs)
    return filtered_obs
