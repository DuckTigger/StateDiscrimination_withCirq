import cirq
import tensorflow as tf
import numpy as np

from cirq_runner import CirqRunner


class BaseModel(CirqRunner):

    def __init__(self, cost_error: tf.Tensor, cost_incon: tf.Tensor,
                 learning_rate: float = 0.5, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-08,
                 noise_on: bool = False, noise_prob: float = 0.1, no_qubits: int = 4, repetitions: int = 100):
        super(CirqRunner, self).__init__(no_qubits=no_qubits, noise_on=noise_on, noise_prob=noise_prob,
                                         sim_repetitions=repetitions)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.Adam_optimiser = tf.optimizers.Adam(self.learning_rate, self.beta1, self.beta2, self.epsilon)
        self.Grad_optimiser = tf.optimizers.SGD(self.learning_rate)
        self.cost_error = cost_error
        self.cost_incon = cost_incon
        self.gate_dict = None
        self.gate_dict_0 = None
        self.gate_dict_1 = None

    @property
    def noise_prob(self):
        return self.__noise_prob

    @noise_prob.setter
    def noise_prob(self, noise_prob):
        self.__noise_prob = noise_prob

    def set_gate_dict(self, gate_dict, measurement_outcome=None):
        if measurement_outcome is None:
            self.gate_dict = gate_dict
        else:
            if measurement_outcome:
                self.gate_dict_1 = gate_dict
            else:
                self.gate_dict_0 = gate_dict

    def state_to_prob(self, state: tf.Tensor) -> tf.Tensor:
        input_state = state.numpy()
        circuit = self.create_full_circuit(self.gate_dict, self.gate_dict_0, self.gate_dict_1)
        probs = self.calculate_probabilities(input_state, circuit)
        return tf.constant(probs)

    def state_to_loss(self, state, label):
        def convert_label(label):
            label = tf.constant([0, 2], dtype=tf.int32, name='pure_labels')
            return label

        def dont_convert(label):
            return tf.cast(label, tf.int32, name='mixed_label')

        label = tf.cond(tf.equal(
            tf.cast(label, tf.int32), 0),
            lambda: convert_label(label),
            lambda: dont_convert(label))

        success = tf