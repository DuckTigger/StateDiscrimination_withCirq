import numpy as np
import tensorflow as tf
import copy
import itertools as it
from typing import Dict, List, Union, Tuple


from tf2_simulator import QSimulator


class TF2SimulatorRunner:
    """
    Will replicate the cirq runner in all necessary functions, except will use the tf simulator as a backend.
    """

    def __init__(self, no_qubits: int = 4, noise_on: bool = False, noise_prob: float = 0.1):
        self.no_qubits = no_qubits
        self.simulator = QSimulator(no_qubits, noise_on, noise_prob)
        self.noise_prob = noise_prob

    @property
    def noise_prob(self):
        return self.__noise_prob

    @noise_prob.setter
    def noise_prob(self, noise_prob):
        if noise_prob > 1:
            self.__noise_prob = 1
        elif noise_prob <= 0:
            self.__noise_prob = 0
        else:
            self.__noise_prob = noise_prob
        self.simulator.noise_prob = noise_prob

    @staticmethod
    def check_density_mat_tf(state: tf.Tensor):
        if not tf.reduce_all(tf.equal(state, tf.math.conj(tf.transpose(state)))):
            return False
        if tf.greater_equal(tf.cast(tf.linalg.trace(state), dtype=tf.float32), tf.constant(1 + 1e-2, dtype=tf.float32))\
                or tf.less_equal(tf.cast(tf.linalg.trace(state), dtype=tf.float32), tf.constant(1 - 1e-2, dtype=tf.float32)):
            return False
        if not tf.reduce_all(tf.greater_equal(tf.cast(tf.linalg.eigvalsh(state), dtype=tf.float32), tf.constant(-1e-8, dtype=tf.float32))):
            return False
        return True

    def probs_controlled_part(self, gate_dict: Dict, state_in: tf.Tensor, prob):
        if tf.greater_equal(tf.cast(prob, dtype=tf.float32), tf.constant(-1e-8, dtype=tf.float32)) and self.check_density_mat_tf(state_in):
            state_out = self.simulator.apply_gate_dict(gate_dict, state_in)
            prob_0, _ = self.simulator.return_prob_and_state(state_out, qid=1, measurement=0)
            prob_1, _ = self.simulator.return_prob_and_state(state_out, qid=1, measurement=1)
        else:
            prob_0 = 0
            prob_1 = 0
        return prob_0, prob_1

    def calculate_probabilities(self, dicts: Tuple[Dict, Dict, Dict], state: tf.Tensor):
        state_pre_measure = self.simulator.apply_gate_dict(dicts[0], state)

        prob_0, state_0 = self.simulator.return_prob_and_state(state_pre_measure, qid=0, measurement=0)
        prob_1, state_1 = self.simulator.return_prob_and_state(state_pre_measure, qid=0, measurement=1)

        prob_0_0, prob_0_1 = self.probs_controlled_part(dicts[1], state_0, prob_0)
        prob_1_0, prob_1_1 = self.probs_controlled_part(dicts[2], state_1, prob_1)

        fin_00 = prob_0 * prob_0_0
        fin_01 = prob_0 * prob_0_1
        fin_10 = prob_1 * prob_1_0
        fin_11 = prob_1 * prob_1_1

        out = [fin_00, fin_01, fin_10, fin_11]
        return out
