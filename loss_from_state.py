import tensorflow as tf

from cirq_runner import CirqRunner


class LossFromState:

    def __init__(self, cost_error: tf.Tensor, cost_incon: tf.Tensor,
                 noise_on: bool = False, noise_prob: float = 0.1, no_qubits: int = 4, repetitions: int = 100):
        self.runner = CirqRunner(no_qubits=no_qubits, noise_on=noise_on, noise_prob=noise_prob,
                                         sim_repetitions=repetitions)
        self.cost_error = cost_error
        self.cost_incon = cost_incon
        self.gate_dict = None
        self.gate_dict_0 = None
        self.gate_dict_1 = None

    def set_all_dicts(self, gate_dict, gate_dict_0, gate_dict_1):
        self.set_gate_dict(gate_dict)
        self.set_gate_dict(gate_dict_0, 0)
        self.set_gate_dict(gate_dict_1, 1)

    def set_gate_dict(self, gate_dict, measurement_outcome=None):
        # Convert between the old style of gate_dicts
        gate_dict['qid'] = gate_dict['qid'] - 1
        gate_dict['control_qid'] = gate_dict['control_qid'] - 1
        if measurement_outcome is None:
            self.gate_dict = gate_dict
        else:
            if measurement_outcome:
                self.gate_dict_1 = gate_dict
            else:
                self.gate_dict_0 = gate_dict

    def state_to_prob(self, state: tf.Tensor) -> tf.Tensor:
        """
        Takes a single input state (in Tensor form) and uses the CirqRunner module to calculate the
        probability of measurements 00, 01, 10, 11.
        :param state: A tensor representing the density matrix of the state to be discriminated.
        :return: prob: A tensor of the measurement probabilities.
        """
        input_state = state.numpy()
        circuit = self.runner.create_full_circuit(self.gate_dict, self.gate_dict_0, self.gate_dict_1)
        probs = self.runner.calculate_probabilities(input_state, circuit)
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

    def state_to_loss(self, state: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
        """
        Chains the two previous functions together to go from a state and label to the loss
        :param state: A tensor representing the incoming state
        :param label: The label of this state (0, 1): pure or mixed
        :return: loss: The loss for this state
        """
        probs = self.state_to_prob(state)
        loss = self.probs_to_loss(probs, label)
        return loss
