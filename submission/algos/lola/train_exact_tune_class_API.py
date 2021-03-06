##########
# Code modified from: https://github.com/alshedivat/lola
##########

import json
import os
import random
import torch
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from ray import tune

from submission.algos.lola.utils import (
    GetFlatWtSess,
    SetFromFlatWtSess,
    flatgrad,
)


class Qnetwork:
    """
    Q-network that is either a look-up table or an MLP with 1 hidden layer.
    """

    def __init__(
        self, myScope, num_hidden, sess, simple_net=True, n_actions=2, std=1.0
    ):
        with tf.variable_scope(myScope):
            # self.input_place = tf.placeholder(shape=[5], dtype=tf.int32)
            # if simple_net:
            #     self.p_act = tf.Variable(tf.random_normal([5, 1], stddev=3.0))
            # else:
            #     act = tf.nn.tanh(
            #         layers.fully_connected(
            #             tf.one_hot(self.input_place, 5, dtype=tf.float32),
            #             num_outputs=num_hidden,
            #             activation_fn=None,
            #         )
            #     )
            #     self.p_act = layers.fully_connected(
            #         act, num_outputs=1, activation_fn=None
            #     )
            self.input_place = tf.placeholder(
                shape=[n_actions ** 2 + 1], dtype=tf.int32
            )
            if simple_net:
                self.p_act = tf.Variable(
                    tf.random_normal(
                        [n_actions ** 2 + 1, n_actions], stddev=std
                    )
                )
            else:
                raise ValueError()
        self.parameters = []
        for i in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=myScope
        ):
            self.parameters.append(i)  # i.name if you want just a name
        self.setparams = SetFromFlatWtSess(self.parameters, sess)
        self.getparams = GetFlatWtSess(self.parameters, sess)


def update(mainQN, lr, final_delta_1_v, final_delta_2_v):
    update_theta_1 = mainQN[0].setparams(
        mainQN[0].getparams() + lr * np.squeeze(final_delta_1_v)
    )
    update_theta_2 = mainQN[1].setparams(
        mainQN[1].getparams() + lr * np.squeeze(final_delta_2_v)
    )


def corrections_func(mainQN, corrections, gamma, pseudo, reg, n_actions=2):
    print_opts = []

    mainQN[0].lr_correction = tf.placeholder(shape=[1], dtype=tf.float32)
    mainQN[1].lr_correction = tf.placeholder(shape=[1], dtype=tf.float32)

    theta_1_all = mainQN[0].p_act
    theta_2_all = mainQN[1].p_act

    # Using sigmoid + normalize to keep values similar to the official
    # implementation
    pi_player_1_for_all_states = tf.nn.sigmoid(theta_1_all)
    pi_player_2_for_all_states = tf.nn.sigmoid(theta_2_all)
    print_opts.append(
        tf.print("pi_player_1_for_all_states", pi_player_1_for_all_states)
    )
    print_opts.append(
        tf.print("pi_player_2_for_all_states", pi_player_2_for_all_states)
    )
    sum_1 = tf.reduce_sum(pi_player_1_for_all_states, axis=1)
    sum_1 = tf.stack([sum_1 for _ in range(n_actions)], axis=1)
    sum_2 = tf.reduce_sum(pi_player_2_for_all_states, axis=1)
    sum_2 = tf.stack([sum_2 for _ in range(n_actions)], axis=1)
    pi_player_1_for_all_states = pi_player_1_for_all_states / sum_1
    pi_player_2_for_all_states = pi_player_2_for_all_states / sum_2
    print_opts.append(
        tf.print("pi_player_1_for_all_states", pi_player_1_for_all_states)
    )
    mainQN[0].policy = pi_player_1_for_all_states
    mainQN[1].policy = pi_player_2_for_all_states
    n_states = int(pi_player_1_for_all_states.shape[0])
    print_opts.append(tf.print("mainQN[0].policy", mainQN[0].policy))
    print_opts.append(tf.print("mainQN[1].policy", mainQN[1].policy))
    pi_player_1_for_states_in_game = tf.slice(
        pi_player_1_for_all_states,
        [0, 0],
        [n_states - 1, n_actions],
    )
    pi_player_2_for_states_in_game = tf.slice(
        pi_player_2_for_all_states,
        [0, 0],
        [n_states - 1, n_actions],
    )
    pi_player_1_for_initial_state = tf.slice(
        pi_player_1_for_all_states,
        [n_states - 1, 0],
        [1, n_actions],
    )
    pi_player_2_for_initial_state = tf.slice(
        pi_player_2_for_all_states,
        [n_states - 1, 0],
        [1, n_actions],
    )
    pi_player_1_for_initial_state = tf.transpose(pi_player_1_for_initial_state)
    pi_player_2_for_initial_state = tf.transpose(pi_player_2_for_initial_state)
    print_opts.append(
        tf.print(
            "pi_player_1_for_initial_state", pi_player_1_for_initial_state
        )
    )

    s_0 = tf.reshape(
        tf.matmul(
            pi_player_1_for_initial_state,
            tf.transpose(pi_player_2_for_initial_state),
        ),
        [-1, 1],
    )

    pi_p1 = tf.reshape(pi_player_1_for_states_in_game, [n_states - 1, -1, 1])
    pi_p2 = tf.reshape(pi_player_2_for_states_in_game, [n_states - 1, -1, 1])

    all_actions_proba_pairs = []
    for action_p1 in range(n_actions):
        for action_p2 in range(n_actions):
            all_actions_proba_pairs.append(
                tf.multiply(pi_p1[:, action_p1], pi_p2[:, action_p2])
            )
    P = tf.concat(
        all_actions_proba_pairs,
        1,
    )
    # if n_actions == 2:
    #     # CC, CD, DC, DD
    #
    #     P = tf.concat(
    #         [
    #             tf.multiply(pi_p1[:, 0], pi_p2[:, 0]),
    #             tf.multiply(pi_p1[:, 0], pi_p2[:, 1]),
    #             tf.multiply(pi_p1[:, 1], pi_p2[:, 0]),
    #             tf.multiply(pi_p1[:, 1], pi_p2[:, 1]),
    #         ],
    #         1,
    #     )
    # elif n_actions == 3:
    #     # CC, CD, CN, DC, DD, DN, NC, ND, NN
    #     P = tf.concat(
    #         [
    #             tf.multiply(pi_p1[:, 0], pi_p2[:, 0]),
    #             tf.multiply(pi_p1[:, 0], pi_p2[:, 1]),
    #             tf.multiply(pi_p1[:, 0], pi_p2[:, 2]),
    #             tf.multiply(pi_p1[:, 1], pi_p2[:, 0]),
    #             tf.multiply(pi_p1[:, 1], pi_p2[:, 1]),
    #             tf.multiply(pi_p1[:, 1], pi_p2[:, 2]),
    #             tf.multiply(pi_p1[:, 2], pi_p2[:, 0]),
    #             tf.multiply(pi_p1[:, 2], pi_p2[:, 1]),
    #             tf.multiply(pi_p1[:, 2], pi_p2[:, 2]),
    #         ],
    #         1,
    #     )
    # else:
    #     raise ValueError(f"n_actions {n_actions}")
    # R_1 = tf.placeholder(shape=[4, 1], dtype=tf.float32)
    # R_2 = tf.placeholder(shape=[4, 1], dtype=tf.float32)
    R_1 = tf.placeholder(shape=[n_actions ** 2, 1], dtype=tf.float32)
    R_2 = tf.placeholder(shape=[n_actions ** 2, 1], dtype=tf.float32)

    # I_m_P = tf.diag([1.0, 1.0, 1.0, 1.0]) - P * gamma
    I_m_P = tf.diag([1.0] * (n_actions ** 2)) - P * gamma
    v_0 = tf.matmul(
        tf.matmul(tf.matrix_inverse(I_m_P), R_1), s_0, transpose_a=True
    )
    v_1 = tf.matmul(
        tf.matmul(tf.matrix_inverse(I_m_P), R_2), s_0, transpose_a=True
    )
    print_opts.append(tf.print("s_0", s_0))
    print_opts.append(tf.print("I_m_P", I_m_P))
    print_opts.append(tf.print("R_1", R_1))
    print_opts.append(tf.print("v_0", v_0))
    if reg > 0:
        for indx, _ in enumerate(mainQN[0].parameters):
            v_0 -= reg * tf.reduce_sum(
                tf.nn.l2_loss(tf.square(mainQN[0].parameters[indx]))
            )
            v_1 -= reg * tf.reduce_sum(
                tf.nn.l2_loss(tf.square(mainQN[1].parameters[indx]))
            )
    v_0_grad_theta_0 = flatgrad(v_0, mainQN[0].parameters)
    v_0_grad_theta_1 = flatgrad(v_0, mainQN[1].parameters)

    v_1_grad_theta_0 = flatgrad(v_1, mainQN[0].parameters)
    v_1_grad_theta_1 = flatgrad(v_1, mainQN[1].parameters)

    v_0_grad_theta_0_wrong = flatgrad(v_0, mainQN[0].parameters)
    v_1_grad_theta_1_wrong = flatgrad(v_1, mainQN[1].parameters)
    param_len = v_0_grad_theta_0_wrong.get_shape()[0].value

    if pseudo:
        multiply0 = tf.matmul(
            tf.reshape(v_0_grad_theta_1, [1, param_len]),
            tf.reshape(v_1_grad_theta_1, [param_len, 1]),
        )
        multiply1 = tf.matmul(
            tf.reshape(v_1_grad_theta_0, [1, param_len]),
            tf.reshape(v_0_grad_theta_0, [param_len, 1]),
        )
    else:
        multiply0 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_0_grad_theta_1), [1, param_len]),
            tf.reshape(v_1_grad_theta_1_wrong, [param_len, 1]),
        )
        multiply1 = tf.matmul(
            tf.reshape(tf.stop_gradient(v_1_grad_theta_0), [1, param_len]),
            tf.reshape(v_0_grad_theta_0_wrong, [param_len, 1]),
        )

    # with tf.control_dependencies(print_opts):
    second_order0 = flatgrad(multiply0, mainQN[0].parameters)
    second_order1 = flatgrad(multiply1, mainQN[1].parameters)

    mainQN[0].R1 = R_1
    mainQN[1].R1 = R_2
    mainQN[0].v = v_0
    mainQN[1].v = v_1
    mainQN[0].delta = v_0_grad_theta_0
    mainQN[1].delta = v_1_grad_theta_1
    mainQN[0].delta += tf.multiply(second_order0, mainQN[0].lr_correction)
    mainQN[1].delta += tf.multiply(second_order1, mainQN[1].lr_correction)


class LOLAExactTrainer(tune.Trainable):
    def _init_lola(
        self,
        *,
        env_name="IteratedAsymBoS",
        num_episodes=50,
        trace_length=200,
        simple_net=True,
        corrections=True,
        pseudo=False,
        num_hidden=10,
        reg=0.0,
        lr=1.0,
        lr_correction=0.5,
        gamma=0.96,
        with_linear_LR_decay_to_zero=False,
        clip_update=None,
        re_init_every_n_epi=1,
        Q_net_std=1.0,
        **kwargs,
    ):

        # print("args not used:", kwargs)

        self.num_episodes = num_episodes
        self.trace_length = trace_length
        self.simple_net = simple_net
        self.corrections = corrections
        self.pseudo = pseudo
        self.num_hidden = num_hidden
        self.reg = reg
        self.lr = lr
        self.lr_correction = lr_correction
        self.gamma = gamma
        self.with_linear_LR_decay_to_zero = with_linear_LR_decay_to_zero
        self.clip_update = clip_update
        self.re_init_every_n_epi = re_init_every_n_epi
        self.Q_net_std = Q_net_std

        graph = tf.Graph()

        with graph.as_default() as g:
            self.sess = tf.Session()
            # Get info about the env

            if env_name == "IteratedPrisonersDilemma":
                self.payout_mat_1 = np.array([[-1.0, 0.0], [-3.0, -2.0]])
                self.payout_mat_2 = self.payout_mat_1.T
            elif env_name == "IteratedAsymBoS":
                self.payout_mat_1 = np.array([[+4.0, 0.0], [0.0, +2.0]])
                self.payout_mat_2 = np.array([[+1.0, 0.0], [0.0, +2.0]])
            elif "custom_payoff_matrix" in kwargs.keys():
                custom_matrix = kwargs["custom_payoff_matrix"]
                self.payout_mat_1 = custom_matrix[:, :, 0]
                self.payout_mat_2 = custom_matrix[:, :, 1]
            elif env_name == "IteratedAsymBoSandPD":

                self.payout_mat_1 = np.array(
                    [
                        [4.0, +0, -3],
                        [+0.0, +2, -3],
                        [+2.0, +2, -1],
                    ]
                )
                self.payout_mat_2 = np.array(
                    [
                        [1, +0, +2],
                        [0, +2, +2],
                        [-3, -3, -1],
                    ]
                )
            else:
                raise ValueError(f"exp_name: {env_name}")
            self.n_actions = int(self.payout_mat_1.shape[0])
            assert self.n_actions == self.payout_mat_1.shape[1]
            self.policy1 = [0.0] * self.n_actions
            self.policy2 = [0.0] * self.n_actions

            # Sanity

            # Q-networks
            self.mainQN = []
            for agent in range(2):
                self.mainQN.append(
                    Qnetwork(
                        "main" + str(agent),
                        self.num_hidden,
                        self.sess,
                        self.simple_net,
                        self.n_actions,
                        self.Q_net_std,
                    )
                )

            # Corrections
            corrections_func(
                self.mainQN,
                self.corrections,
                self.gamma,
                self.pseudo,
                self.reg,
                self.n_actions,
            )

            self.results = []
            self.norm = 1 / (1 - self.gamma)
            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            # + tf.get_collection_ref("batch_norm_non_trainable_variables_co??????llection")

    # TODO add something to not load and create everything when only evaluating with RLLib

    def setup(self, config):
        self._init_lola(**config)

    def step(self):

        log_items = {}
        log_items["episode"] = self.training_iteration

        if self.training_iteration % self.re_init_every_n_epi == 0:
            self.sess.run(self.init)

        lr_coor = np.ones(1) * self.lr_correction

        res = []
        params_time = []
        delta_time = []
        input_vals = self._get_input_vals()
        for i in range(self.trace_length):
            params0 = self.mainQN[0].getparams()
            params1 = self.mainQN[1].getparams()
            outputs = [
                self.mainQN[0].delta,
                self.mainQN[1].delta,
                self.mainQN[0].v,
                self.mainQN[1].v,
                self.mainQN[0].policy,
                self.mainQN[1].policy,
            ]
            (update1, update2, v1, v2, policy1, policy2) = self.sess.run(
                outputs,
                feed_dict={
                    self.mainQN[0].input_place: input_vals,
                    self.mainQN[1].input_place: input_vals,
                    self.mainQN[0].R1: np.reshape(self.payout_mat_1, [-1, 1]),
                    self.mainQN[1].R1: np.reshape(self.payout_mat_2, [-1, 1]),
                    self.mainQN[0].lr_correction: lr_coor,
                    self.mainQN[1].lr_correction: lr_coor,
                },
            )
            update1, update2 = self._clip_update(update1, update2)
            self._update_with_lr_decay(update1, update2, i)
            params_time.append([params0, params1])
            delta_time.append([update1, update2])

            log_items["ret1"] = v1[0][0] / self.norm
            log_items["ret2"] = v2[0][0] / self.norm
            res.append([v1[0][0] / self.norm, v2[0][0] / self.norm])
        self.results.append(res)

        if self.training_iteration % self.re_init_every_n_epi == (
            self.re_init_every_n_epi - 1
        ):
            self.policy1 = policy1
            self.policy2 = policy2

        log_items["episodes_total"] = self.training_iteration
        log_items["policy1"] = self.policy1
        log_items["policy2"] = self.policy2

        return log_items

    def _get_input_vals(self):
        return np.reshape(np.array(range(self.n_actions ** 2 + 1)) + 1, [-1])

    def _clip_update(self, update1, update2):
        if self.clip_update is not None:
            assert self.clip_update > 0.0
            update1_norm = np.linalg.norm(update1)
            if update1_norm > self.clip_update:
                multiplier = self.clip_update / update1_norm
                update1 *= multiplier
            update2_norm = np.linalg.norm(update2)
            if update2_norm > self.clip_update:
                multiplier = self.clip_update / update2_norm
                update2 *= multiplier
            # update1 = np.clip(update1, a_min=-self.clip_update, a_max=self.clip_update)
            # update2 = np.clip(update2, a_min=-self.clip_update, a_max=self.clip_update)
        return update1, update2

    def _update_with_lr_decay(self, update1, update2, i):
        if self.with_linear_LR_decay_to_zero:
            n_step_from_start = self.training_iteration * self.trace_length + i
            n_total_steps = (self.num_episodes + 1) * self.trace_length
            decayed_lr = self.lr * (1 - (n_step_from_start / n_total_steps))
            assert decayed_lr >= 0.0
            update(self.mainQN, decayed_lr, update1, update2)
        else:
            update(self.mainQN, self.lr, update1, update2)

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint.json")
        tf_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        tf_checkpoint_dir, tf_checkpoint_filename = os.path.split(
            tf_checkpoint_path
        )
        checkpoint = {
            "timestep": self.training_iteration,
            "tf_checkpoint_dir": tf_checkpoint_dir,
            "tf_checkpoint_filename": tf_checkpoint_filename,
        }
        with open(path, "w") as f:
            json.dump(checkpoint, f, sort_keys=True, indent=4)

        # TF v1
        save_path = self.saver.save(self.sess, f"{tf_checkpoint_path}.ckpt")

        return path

    def load_checkpoint(self, checkpoint_path):

        checkpoint_path = os.path.expanduser(checkpoint_path)
        print("Loading Model...", checkpoint_path)
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        print("checkpoint", checkpoint)

        # Support VM and local (manual) loading
        tf_checkpoint_dir, _ = os.path.split(checkpoint_path)
        print("tf_checkpoint_dir", tf_checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(
            tf_checkpoint_dir,
            latest_filename=f'{checkpoint["tf_checkpoint_filename"]}',
        )
        tail, head = os.path.split(ckpt.model_checkpoint_path)
        ckpt.model_checkpoint_path = os.path.join(tf_checkpoint_dir, head)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def cleanup(self):
        self.sess.close()
        super().cleanup()

    def _get_agent_to_use(self, policy_id):
        if policy_id == "player_red":
            agent_n = 0
        elif policy_id == "player_blue":
            agent_n = 1
        elif policy_id == "player_row":
            agent_n = 0
        elif policy_id == "player_col":
            agent_n = 1
        else:
            raise ValueError(f"policy_id {policy_id}")
        return agent_n

    def _preprocess_obs(self, single_obs, agent_to_use):
        single_obs = np.where(single_obs == 1)[0][0]

        return single_obs

    def _post_process_action(self, action):
        return action[None, ...]  # add batch dim

    def compute_actions(self, policy_id: str, obs_batch: list):
        assert (
            len(obs_batch) == 1
        ), f"{len(obs_batch)} == 1. obs_batch: {obs_batch}"

        for single_obs in obs_batch:
            agent_to_use = self._get_agent_to_use(policy_id)
            obs = self._preprocess_obs(single_obs, agent_to_use)
            input_vals = self._get_input_vals()
            policy = self.sess.run(
                [
                    self.mainQN[agent_to_use].policy,
                ],
                feed_dict={
                    self.mainQN[agent_to_use].input_place: input_vals,
                },
            )
            probabilities = policy[0][obs]
            probabilities = torch.tensor(probabilities)
            policy_for_this_state = torch.distributions.Categorical(
                probs=probabilities
            )
            action = policy_for_this_state.sample()

        action = self._post_process_action(action)

        state_out = []
        extra_fetches = {}
        return action, state_out, extra_fetches
