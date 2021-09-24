from ray import tune

from submission.envs.matrix_sequential_social_dilemma import (
    IteratedPrisonersDilemma,
)
from submission.utils import log


def get_rllib_config(seeds, debug=False):
    stop_config = {
        "episodes_total": 2 if debug else 400,
    }

    n_steps_in_epi = 20

    env_config = {
        "players_ids": ["player_row", "player_col"],
        "max_steps": n_steps_in_epi,
        "get_additional_info": True,
    }

    rllib_config = {
        "env": IteratedPrisonersDilemma,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    None,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    {},
                ),
                env_config["players_ids"][1]: (
                    None,
                    IteratedPrisonersDilemma.OBSERVATION_SPACE,
                    IteratedPrisonersDilemma.ACTION_SPACE,
                    {},
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
        },
        "seed": tune.grid_search(seeds),
        "callbacks": log.get_logging_callbacks_class(log_full_epi=True),
        "framework": "torch",
        "rollout_fragment_length": n_steps_in_epi,
        "train_batch_size": n_steps_in_epi,
    }

    return rllib_config, stop_config
