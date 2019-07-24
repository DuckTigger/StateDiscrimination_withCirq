import numpy as np
import cirq
from typing import Counter
import tensorflow as tf
import copy

from tf2_simulator_runner import TF2SimulatorRunner
from cirq_runner import CirqRunner
from gate_dictionaries import GateDictionaries
from test.test_model import TestLossFromState


class TestTFRunner(np.testing.TestCase):

    @staticmethod
    def kron_list(l):
        for i in range(len(l)):
            if i == 0:
                out = l[i]
            else:
                out = np.kron(out, l[i])
        return out

    def test_calculate_probabilities(self):
        cirq_runner = CirqRunner(sim_repetitions=1000)
        qubits = cirq_runner.qubits
        zero = np.array([[1, 0], [0, 0]])
        z_list = [zero for _ in range(4)]
        zero_state = self.kron_list(z_list).astype(np.complex64)
        z_copy_0 = copy.copy(zero_state)
        z_copy_1 = copy.copy(zero_state)
        z_copy_2 = copy.copy(zero_state)
        circuit = cirq.Circuit.from_ops([cirq.X(x) for x in qubits])
        circuit.append(cirq.measure(qubits[0], key='m0'))
        circuit.append(cirq.Circuit.from_ops([cirq.X(x) for x in qubits]))
        circuit.append(cirq.measure(qubits[1], key='m1'))
        probs0 = cirq_runner.calculate_probabilities_sampling(zero_state, circuit)
        gate_dicts = GateDictionaries.build_dict(np.array([1, 1, 1, 1]), np.array([]), np.array([0,1,2,3]),
                                                 np.array([np.pi, np.pi, np.pi, np.pi]))
        gate_dicts_post = GateDictionaries.build_dict(np.array([4, 1, 1, 1]), np.array([]), np.array([0,1,2,3]),
                                                 np.array([np.pi, np.pi, np.pi]))
        tf_runner = TF2SimulatorRunner()
        dicts = (gate_dicts, gate_dicts_post, gate_dicts_post)
        probs_tf = tf_runner.calculate_probabilities(dicts, tf.constant(z_copy_0))
        np.testing.assert_array_equal(probs0, [0, 0, 1, 0])
        np.testing.assert_almost_equal(probs_tf, probs0, decimal=1)

        one = np.array([[0, 0], [0, 1]])
        o_list = [one for _ in range(4)]
        one_state = self.kron_list(o_list).astype(np.complex64)
        o_copy = copy.copy(one_state)
        probs1 = cirq_runner.calculate_probabilities_sampling(one_state, circuit)
        probs_1tf = tf_runner.calculate_probabilities(dicts, tf.constant(o_copy))
        np.testing.assert_array_equal(probs1, [0, 1, 0, 0])
        np.testing.assert_almost_equal(probs_1tf, probs1, decimal=1)

        circuit2 = cirq.Circuit.from_ops([cirq.H(x) for x in qubits])
        gate_dicts = GateDictionaries.build_dict(np.array([5, 5, 5, 5]), np.array([]),np.array([0,1,2,3]))
        gate_dicts_post = GateDictionaries.build_dict(np.array([4, 4, 4, 4]), np.array([]), np.array([0, 1, 2, 3]))
        dicts = (gate_dicts, gate_dicts_post, gate_dicts_post)
        circuit2.append(cirq.measure(qubits[0], key='m0'))
        circuit2.append(cirq.measure(qubits[1], key='m1'))
        probs2 = cirq_runner.calculate_probabilities_sampling(z_copy_1, circuit2)
        probs_tf_2 = tf_runner.calculate_probabilities(dicts, tf.constant(z_copy_2))
        np.testing.assert_array_almost_equal(probs2, [0.25, 0.25, 0.25, 0.25], decimal=2)
        np.testing.assert_array_almost_equal(probs_tf_2, probs2, decimal=1)

    def test_compare_prob_methods(self):
        c_runner = CirqRunner(sim_repetitions=10000)
        tf_runner = TF2SimulatorRunner()
        dicts = TestLossFromState.get_some_dicts()
        for gate_dict in dicts:
            gate_dict['theta'] = [tf.Variable(x) for x in gate_dict['theta']]
        zero, _ = TestLossFromState.get_some_states()
        zero_ = copy.copy(zero)
        zero__ = copy.copy(zero)
        circuit_a = c_runner.create_full_circuit(dicts[0], dicts[1], dicts[2])
        probs_a = c_runner.calculate_probabilities_sampling(zero, circuit_a)
        for d in dicts:
            for key, val in d.items():
                if key == 'gate_id':
                    d[key] = np.append(d[key], 4)
                elif key == 'qid':
                    d[key] = np.append(d[key], 0)

        probs_b = c_runner.calculate_probabilities((dicts[0], dicts[1], dicts[2]), zero_)
        probs_c = tf_runner.calculate_probabilities(dicts, zero__)
        np.testing.assert_almost_equal(np.sum(probs_a), 1)
        np.testing.assert_almost_equal(np.sum(probs_c), 1)
        np.testing.assert_array_almost_equal(probs_a, probs_b, decimal=4)
        np.testing.assert_array_almost_equal(probs_c, probs_b, decimal=1)

