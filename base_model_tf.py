import tensorflow as tf
import numpy as np
import copy
from typing import List

from tf2_simulator_runner import TF2SimulatorRunner


class ModelTF(tf.keras.Model):

    def __init__(self, cost_error: float, cost_incon: float, runner: TF2SimulatorRunner):
        super().__init__()
        self.runner = runner
        self.cost_error = tf.constant(cost_error, dtype=tf.float64)
        self.cost_incon = tf.constant(cost_incon, dtype=tf.float64)
        self.gate_dict = None
        self.gate_dict_0 = None
        self.gate_dict_1 = None

    @property
    def runner(self):
        return self.__runner

    @runner.setter
    def runner(self, runner: TF2SimulatorRunner):
        self.__runner = runner

    def set_all_dicts(self, gate_dict, gate_dict_0, gate_dict_1):
        self.set_gate_dict(gate_dict)
        self.set_gate_dict(gate_dict_0, 0)
        self.set_gate_dict(gate_dict_1, 1)

    def set_gate_dict(self, gate_dict, measurement_outcome=None):
        if measurement_outcome is None:
            self.gate_dict = gate_dict
        else:
            if measurement_outcome:
                self.gate_dict_1 = gate_dict
            else:
                self.gate_dict_0 = gate_dict

    def return_gate_dicts(self):
        return self.gate_dict, self.gate_dict_0, self.gate_dict_1

    def no_of_variabes(self):
        gate_dict, gate_dict_0, gate_dict_1 = self.return_gate_dicts()
        return len(gate_dict['theta_indices']) + len(gate_dict_0['theta_indices']) + len(gate_dict_1['theta_indices'])

    def set_variables(self, variables: List[tf.Variable]):
        gate_dict, gate_dict_0, gate_dict_1 = self.return_gate_dicts()
        vars0 = len(gate_dict['theta_indices'])
        vars1 = len(gate_dict_0['theta_indices']) + vars0
        vars2 = len(gate_dict_1['theta_indices']) + vars1

        self.gate_dict['theta'] = [x for x in variables[:vars0]]
        self.gate_dict_0['theta'] = [x for x in variables[vars0:vars1]]
        self.gate_dict_1['theta'] = [x for x in variables[vars1:vars2]]

    def get_variables(self):
        gate_dict, gate_dict_0, gate_dict_1 = self.return_gate_dicts()
        variables = gate_dict['theta'] + gate_dict_0['theta'] + gate_dict_1['theta']
        return variables

    def get_gate_ids(self):
        gate_dict, gate_dict_0, gate_dict_1 = self.return_gate_dicts()
        gate_ids = gate_dict['gate_id'] + gate_dict_0['gate_id'] + gate_dict_1['gate_id']
        gate_ids = gate_ids[np.where(gate_ids != 0)]
        return gate_ids

    # @tf.custom_gradient
    def loss_fn(self, state_in: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """
        Takes a tensor of a circuit's measurement probabilities and returns the loss for this circuit, given the label
        of the input state.
        :param probs: A tensor of measurement probabilities
        :param label: The label of the state - defining success / error - converted into measurements in the functions
        below.
        :return: loss: a tensor representing the loss associated with this state
        """

        probs = self.runner.calculate_probabilities(self.return_gate_dicts(), state_in)

        def convert_label(label):
            label = tf.constant([0, 2], dtype=tf.int32, name='pure_labels')
            return label

        def dont_convert(label):
            return tf.cast(label, tf.int32, name='mixed_label')

        label = tf.cond(tf.equal(
            tf.cast(label, tf.int32), 0),
            lambda: convert_label(label),
            lambda: dont_convert(label))

        loss = self.loss(probs, label)
        return loss

    def loss(self, probs: tf.Tensor, label: tf.Tensor):
        success = tf.reduce_sum(tf.gather(probs, label))
        inconclusive = tf.multiply(tf.gather(probs, 3), self.cost_incon)
        error = tf.multiply(tf.subtract(1, tf.add(success, inconclusive)), self.cost_error)
        loss = tf.reduce_sum([error, inconclusive])
        return loss

    def variables_gradient_exact(self, state: np.ndarray, label: tf.Tensor) -> List:
        """
        Calculates the gradient of the loss function w.r.t. each variable, for a small change in variable defined
        by g_epsilon.
        :param state: The state in
        :param label: the label of that state
        :return: grads: a list of tensors representing the gradients for each variable.
        """
        variables = self.get_variables()
        grads = []
        for i, var in enumerate(variables):
            new_vars_plus = copy.copy(variables)
            new_vars_minus = copy.copy(variables)

            new_vars_plus[i] = tf.add(var, np.pi/4)
            new_vars_minus[i] = tf.subtract(var, np.pi/4)

            self.set_variables(new_vars_plus)
            loss_plus = self.state_to_loss(state, label)

            self.set_variables(new_vars_minus)
            loss_minus = self.state_to_loss(state, label)
            grad = tf.subtract(loss_plus, loss_minus)
            grad = tf.reshape(grad, (1,))
            grads.append(grad)

        self.set_variables(variables)
        return grads
