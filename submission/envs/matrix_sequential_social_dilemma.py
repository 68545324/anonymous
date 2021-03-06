##########
# Part of the code modified from:
# https://github.com/alshedivat/lola/tree/master/lola
##########
import logging
from abc import ABC
from collections import Iterable
from typing import Dict

import numpy as np
from gym.spaces import Discrete
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from submission.envs.utils.interfaces import InfoAccumulationInterface
from submission.envs.utils.mixins import (
    TwoPlayersTwoActionsInfoMixin,
    NPlayersNDiscreteActionsInfoMixin,
)

logger = logging.getLogger(__name__)

PLOT_KEYS = [
    "CC_freq",
    "DD_freq",
    "CD_freq",
    "DC_freq",
]

PLOT_ASSEMBLAGE_TAGS = [
    ("_freq_player_row_mean", "_freq_player_col_mean"),
    ("_freq",),
    ("CC_freq",),
    ("DD_freq",),
    ("CD_freq",),
    ("DC_freq",),
]


class MatrixSequentialSocialDilemma(
    InfoAccumulationInterface, MultiAgentEnv, ABC
):
    """
    A multi-agent abstract class for two player matrix games.

    PAYOFF_MATRIX: Numpy array. Along the dimension N, the action of the
    Nth player change. The last dimension is used to select the player
    whose reward you want to know.

    max_steps: number of step in one episode

    players_ids: list of the RLLib agent id of each player

    output_additional_info: ask the environment to aggregate information
    about the last episode and output them as info at the end of the
    episode.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = None
    NUM_STATES = None
    ACTION_SPACE = None
    OBSERVATION_SPACE = None
    PAYOFF_MATRIX = None
    NAME = None

    def __init__(self, config: Dict = {}):

        assert self.PAYOFF_MATRIX is not None
        assert self.PAYOFF_MATRIX.shape[0] == self.NUM_ACTIONS
        assert self.PAYOFF_MATRIX.shape[1] == self.NUM_ACTIONS
        assert self.PAYOFF_MATRIX.shape[2] == self.NUM_AGENTS
        assert len(self.PAYOFF_MATRIX.shape) == 3
        if "players_ids" in config:
            assert (
                isinstance(config["players_ids"], Iterable)
                and len(config["players_ids"]) == self.NUM_AGENTS
            )

        self.players_ids = config.get(
            "players_ids", ["player_row", "player_col"]
        )
        self.player_row_id, self.player_col_id = self.players_ids
        self.max_steps = config.get("max_steps", 20)
        self.output_additional_info = config.get(
            "output_additional_info", True
        )
        self.same_obs_for_each_player = config.get(
            "same_obs_for_each_player", True
        )

        self.step_count_in_current_episode = None

        # To store info about the fraction of each states
        if self.output_additional_info:
            self._init_info()

    def seed(self, seed=None):
        """Seed the PRNG of this space."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_count_in_current_episode = 0
        if self.output_additional_info:
            self._reset_info()
        return {
            self.player_row_id: self.NUM_STATES - 1,
            self.player_col_id: self.NUM_STATES - 1,
        }

    def step(self, actions: dict):
        """
        :param actions: Dict containing both actions for player_1 and player_2
        :return: observations, rewards, done, info
        """
        self.step_count_in_current_episode += 1
        action_player_row = actions[self.player_row_id]
        action_player_col = actions[self.player_col_id]

        if self.output_additional_info:
            self._accumulate_info(action_player_row, action_player_col)

        observations = self._produce_observations(
            action_player_row, action_player_col
        )
        rewards = self._get_players_rewards(
            action_player_row, action_player_col
        )
        epi_is_done = self.step_count_in_current_episode >= self.max_steps
        if self.step_count_in_current_episode > self.max_steps:
            logger.warning(
                "self.step_count_in_current_episode >= self.max_steps"
            )
        info = self._get_info_for_current_epi(epi_is_done)
        return self._to_RLLib_API(observations, rewards, epi_is_done, info)

    def _produce_observations(self, action_player_row, action_player_col):
        if self.same_obs_for_each_player:
            return self._produce_same_observations_for_each_player(
                action_player_row, action_player_col
            )
        else:
            return self._produce_observations_invariant_to_the_player_trained(
                action_player_row, action_player_col
            )

    def _produce_same_observations_for_each_player(
        self, action_player_0: int, action_player_1: int
    ):
        return [
            action_player_0 * self.NUM_ACTIONS + action_player_1,
            action_player_0 * self.NUM_ACTIONS + action_player_1,
        ]

    def _produce_observations_invariant_to_the_player_trained(
        self, action_player_0: int, action_player_1: int
    ):
        """
        We want to be able to use a policy trained as player 1
        for evaluation as player 2 and vice versa.
        """
        return [
            action_player_0 * self.NUM_ACTIONS + action_player_1,
            action_player_1 * self.NUM_ACTIONS + action_player_0,
        ]

    def _get_players_rewards(self, action_player_0: int, action_player_1: int):
        return [
            self.PAYOFF_MATRIX[action_player_0][action_player_1][0],
            self.PAYOFF_MATRIX[action_player_0][action_player_1][1],
        ]

    def _to_RLLib_API(
        self, observations: list, rewards: list, epi_is_done: bool, info: dict
    ):

        observations = {
            self.player_row_id: observations[0],
            self.player_col_id: observations[1],
        }

        rewards = {
            self.player_row_id: rewards[0],
            self.player_col_id: rewards[1],
        }

        if info is None:
            info = {}
        else:
            info = {self.player_row_id: info, self.player_col_id: info}

        done = {
            self.player_row_id: epi_is_done,
            self.player_col_id: epi_is_done,
            "__all__": epi_is_done,
        }

        return observations, rewards, done, info

    def _get_info_for_current_epi(self, epi_is_done):
        if epi_is_done and self.output_additional_info:
            info_for_current_epi = self._get_episode_info()
        else:
            info_for_current_epi = None
        return info_for_current_epi

    def __str__(self):
        return self.NAME


class IteratedMatchingPennies(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Matching Pennies game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[+1, -1], [-1, +1]], [[-1, +1], [+1, -1]]])
    NAME = "IMP"


class IteratedPrisonersDilemma(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Prisoner's Dilemma game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[-1, -1], [-3, +0]], [[+0, -3], [-2, -2]]])
    NAME = "IPD"


class IteratedAsymPrisonersDilemma(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Asymmetric Prisoner's Dilemma game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[+0, -1], [-3, +0]], [[+0, -3], [-2, -2]]])
    NAME = "IPD"


class IteratedStagHunt(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Stag Hunt game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array([[[3, 3], [0, 2]], [[2, 0], [1, 1]]])
    NAME = "IteratedStagHunt"


class IteratedChicken(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Chicken game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [[[+0, +0], [-1.0, +1.0]], [[+1, -1], [-10, -10]]]
    )
    NAME = "IteratedChicken"


class IteratedAsymChicken(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the Asymmetric Chicken game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [[[+2.0, +0], [-1.0, +1.0]], [[+2.5, -1], [-10, -10]]]
    )
    NAME = "AsymmetricIteratedChicken"


class IteratedBoS(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the BoS game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [[[+3.0, +2.0], [+0.0, +0.0]], [[+0.0, +0.0], [+2.0, +3.0]]]
    )
    NAME = "IteratedBoS"


class IteratedAsymBoS(
    TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the BoS game.
    """

    NUM_ACTIONS = 2
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [[[+4.0, +1.0], [+0.0, +0.0]], [[+0.0, +0.0], [+2.0, +2.0]]]
        # [[[+8.0, -2.0], [+0.0, +0.0]], [[+0.0, +0.0], [+2.0, +2.0]]]
    )
    NAME = "IteratedAsymBoS"


def define_greed_fear_matrix_game(greed, fear):
    class GreedFearGame(
        TwoPlayersTwoActionsInfoMixin, MatrixSequentialSocialDilemma
    ):
        NUM_ACTIONS = 2
        NUM_STATES = (
            NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
        )
        ACTION_SPACE = Discrete(NUM_ACTIONS)
        OBSERVATION_SPACE = Discrete(NUM_STATES)
        R = 3
        P = 1
        T = R + greed
        S = P - fear
        PAYOFF_MATRIX = np.array([[[R, R], [S, T]], [[T, S], [P, P]]])
        NAME = "IteratedGreedFear"

        def __str__(self):
            return f"{self.NAME} with greed={greed} and fear={fear}"

    return GreedFearGame


class IteratedBoSAndPD(
    NPlayersNDiscreteActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the BOTS + PD game.
    """

    NUM_ACTIONS = 3
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [
            [[3.5, +1], [+0, +0], [-3, +2]],
            [[+0.0, +0], [+1, +3], [-3, +2]],
            [[+2.0, -3], [+2, -3], [-1, -1]],
        ]
    )
    NAME = "IteratedBoSAndPD"


class IteratedAsymBoSandPD(
    NPlayersNDiscreteActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the BOTS + PD game.
    """

    NUM_ACTIONS = 3
    NUM_STATES = NUM_ACTIONS ** MatrixSequentialSocialDilemma.NUM_AGENTS + 1
    ACTION_SPACE = Discrete(NUM_ACTIONS)
    OBSERVATION_SPACE = Discrete(NUM_STATES)
    PAYOFF_MATRIX = np.array(
        [
            [[+4.0, +1], [0.0, 0.0], [-6.0, +5]],
            [[+0.0, 0.0], [+2.0, +2], [-6.0, +5]],
            [[+5.0, -6], [+5.0, -6], [-1, -1]],
        ]
    )
    NAME = "IteratedAsymBoSandPD"


class TwoPlayersCustomizableMatrixGame(
    NPlayersNDiscreteActionsInfoMixin, MatrixSequentialSocialDilemma
):
    """
    A two-agent environment for the BOTS + PD game.
    """

    NAME = "TwoPlayersCustomizableMatrixGame"

    NUM_ACTIONS = None
    NUM_STATES = None
    ACTION_SPACE = None
    OBSERVATION_SPACE = None
    PAYOFF_MATRIX = None

    def __init__(self, config: Dict):
        self.PAYOFF_MATRIX = config["PAYOFF_MATRIX"]
        self.NUM_ACTIONS = config["NUM_ACTIONS"]
        self.ACTION_SPACE = Discrete(self.NUM_ACTIONS)
        self.NUM_STATES = self.NUM_ACTIONS ** self.NUM_AGENTS + 1
        self.OBSERVATION_SPACE = Discrete(self.NUM_STATES)

        super().__init__(config)
