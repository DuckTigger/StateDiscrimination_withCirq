import tensorflow as tf
import numpy as np
import copy
from typing import List, Union

from cirq_runner import CirqRunner


class Model(tf.keras.Model):

    def __init__(self, cost_error: float, cost_incon: float, runner: CirqRunner,
                 g_epsilon: float = 1e-6):
        super().__init__()
        self.runner = runner
        self.cost_error = tf.constant(cost_error, dtype=tf.float64)
        self.cost_incon = tf.constant(cost_incon, dtype=tf.float64)
        self.g_epsilon = tf.constant(g_epsilon, dtype=tf.float64)
        self.gate_dict = None
        self.gate_dict_0 = None
        self.gate_dict_1 = None

    @property
    def runner(self):
        return self.__runner

    @runner.setter
    def runner(self, runner: CirqRunner):
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
        gate_ids = np.append(np.append(gate_dict['gate_id'], gate_dict_0['gate_id']), gate_dict_1['gate_id'])
        gate_ids = gate_ids[np.where(gate_ids != 0)]
        return gate_ids

    def state_to_prob(self, state: np.ndarray) -> tf.Tensor:
        """
        Takes a single input state (in Tensor form) and uses the CirqRunner module to calculate the
        probability of measurements 00, 01, 10, 11.
        :param state: A tensor representing the density matrix of the state to be discriminated.
        :return: prob: A tensor of the measurement probabilities.
        """
        probs = self.runner.calculate_probabilities((self.gate_dict,
                                                    self.gate_dict_0, self.gate_dict_1), state)
        return tf.constant(probs)

    def probs_to_loss(self, probs: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """
        Takes a tensor of a circuit's measurement probabilities and returns the loss for this circuit, given the label
        of the input state.
        :param probs: A tensor of measurement probabilities
        :param label: The label of the state - defining success / error - converted into measurements in the functions
        below.
        :return: loss: a tensor representing the loss associated with this state
        """
        def convert_label(label):
            label = tf.constant([0, 2], dtype=tf.int32, name='pure_labels')
            return label

        def dont_convert(label):
            return tf.cast(label, tf.int32, name='mixed_label')

        label = tf.cond(tf.equal(
            tf.cast(label, tf.int32), 0),
            lambda: convert_label(label),
            lambda: dont_convert(label))

        success = tf.reduce_sum(tf.gather(probs, label))
        inconclusive = tf.multiply(tf.gather(probs, 3), self.cost_incon)
        error = tf.multiply(tf.subtract(1, tf.add(success, inconclusive)), self.cost_error)
        loss = tf.reduce_sum([error, inconclusive])
        return loss

    def state_to_loss(self, state: np.ndarray, label: tf.Tensor) -> tf.Tensor:
        """
        Chains the two previous functions together to go from a state and label to the loss
        :param state: A tensor representing the incoming state
        :param label: The label of this state (0, 1): pure or mixed
        :return: loss: The loss for this state
        """
        probs = self.state_to_prob(state)
        loss = self.probs_to_loss(probs, label)
        return loss

    def variables_gradient(self, loss: tf.Tensor, state: np.ndarray, label: tf.Tensor) -> List:
        """
        Calculates the gradient of the loss function w.r.t. each variable, for a small change in variable defined
        by g_epsilon.
        :param loss: The current loss of the model
        :param state: The state in
        :param label: the label of that state
        :return: grads: a list of tensors representing the gradients for each variable.
        """
        variables = self.get_variables()
        losses = []
        for i, var in enumerate(variables):
            new_vars = copy.copy(variables)
            new_vars[i] = tf.add(var, self.g_epsilon)
            self.set_variables(new_vars)
            new_loss = self.state_to_loss(state, label)
            losses.append(new_loss)

        self.set_variables(variables)
        dy = [tf.subtract(x, loss) for x in losses]
        grads = [tf.divide([y], self.g_epsilon) for y in dy]
        return grads

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
            grads.append(grad)

        self.set_variables(variables)
        return grads
