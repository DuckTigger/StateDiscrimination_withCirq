import copy
import tensorflow as tf
import numpy as np
from typing import List

from cirq_runner import CirqRunner


class MinimizeEnergyModel(tf.keras.Model):


    def __init__(self, u: float, v: float):
        self.u = u
        self.v = v
        self.runner = CirqRunner(no_qubits=2, sim_repetitions=1000)
        self.gate_dict = None

    def set_gate_dict(self, gate_dict):
        self.gate_dict = gate_dict

    def get_variables(self):
        return self.gate_dict['theta']

    def set_variables(self, variables: List[tf.Variable]):
        gate_dict = self.gate_dict
        vars0 = len(gate_dict['theta_indices'])
        self.gate_dict['theta'] = [x for x in variables[:vars0]]

    def return_gate_dicts(self):
        return self.gate_dict, self.gate_dict_0, self.gate_dict_1

    def energy_fn(self):
        energy = self.runner.calculate_energy_sampling(self.u, self.v, self.gate_dict)
        return energy

    def gradient_fn(self):
        trainable = self.get_variables()
        grads = []

        for i, var in enumerate(trainable):
            v_plus = copy.copy(trainable)
            v_minus = copy.copy(trainable)

            v_plus[i] = tf.add(var, np.pi/4)
            v_minus[i] = tf.subtract(var, np.pi/4)

            self.set_variables(v_plus)
            energy_plus = self.energy_fn()

            self.set_variables(v_minus)
            energy_minus = self.energy_fn()

            grad = tf.subtract(energy_plus, energy_minus)
            grads.append(grad)
        self.set_variables(trainable)
        return grads
