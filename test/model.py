import tensorflow as tf
from typing import List
import copy
import numpy as np
import cirq

from loss_from_state import LossFromState

class Model:

    def __init__(self, cost_error: tf.Tensor, cost_incon: tf.Tensor,
                 noise_on: bool = False, noise_prob: float = 0.1, no_qubits: int = 4, repetitions: int = 100,
                 g_epsilon: tf.Tensor = tf.constant(1e-6)):
        self.loss_calc = LossFromState(cost_error, cost_incon,noise_on, noise_prob, no_qubits, repetitions)
        self.runner = self.loss_calc.runner
        self.qubits = self.runner.get_qubits()
        self.g_epsilon = g_epsilon
        self.cost_error = cost_error
        self.cost_incon = cost_incon

    def variable_gradient(self, loss: tf.Tensor, variables: List[tf.Variable], state: tf.Tensor, label: tf.Tensor):
        for i, var in enumerate(variables):
            new_vars = copy.copy(variables)
            new_vars[i] = tf.add(var, self.g_epsilon)
            
