import numpy as np
import cirq
from typing import Counter
import tensorflow as tf
import copy

from cirq_runner import CirqRunner
from test.test_model import TestLossFromState


class TestCirqRunner(np.testing.TestCase):

    def test_gate_dict_to_circuit(self):
        runner = CirqRunner()
        qubits = runner.get_qubits()
        gate_dict = {
            'gate_id': np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0]),
            'theta': None,
            'theta_indices': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            'control_qid': np.array([0, 1, 2, 3]),
            'control_indices': np.array([12, 13, 14, 15]),
            'qid': np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0])
        }
        theta = [tf.Variable(np.random.rand() * 4*np.pi) for _ in range(len(gate_dict['theta_indices']))]
        gate_dict['theta'] = theta
        circuit = runner.gate_dict_to_circuit(gate_dict)
        simulator = cirq.Simulator()
        circuit.append(cirq.measure(*[x for x in qubits], key='m'))
        result = simulator.run(circuit, repetitions=100)
        hist = result.histogram(key='m')
        self.assertIsInstance(hist, Counter)

    def test_create_full_circuit(self):
        runner = CirqRunner()
        gate_dict = {
            'gate_id': np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0]),
            'theta': None,
            'theta_indices': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            'control_qid': np.array([0, 1, 2, 3]),
            'control_indices': np.array([12, 13, 14, 15]),
            'qid': np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0])
        }
        theta = [tf.Variable(np.random.rand() * 4 * np.pi) for _ in range(len(gate_dict['theta_indices']))]
        gate_dict['theta'] = theta

        gate_dict_0 = {
            'gate_id': np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0]),
            'theta': None,
            'theta_indices': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            'control_qid': np.array([1, 2, 3]),
            'control_indices': np.array([9, 10, 11]),
            'qid': np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1])
        }
        theta0 = [tf.Variable(np.random.rand() * 4 * np.pi) for _ in range(len(gate_dict_0['theta_indices']))]
        gate_dict_0['theta'] = theta0

        gate_dict_1 = {
            'gate_id': np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0]),
            'theta': None,
            'theta_indices': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            'control_qid': np.array([1, 2, 3]),
            'control_indices': np.array([9, 10, 11]),
            'qid': np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1])
        }
        theta1 = [tf.Variable(np.random.rand() * 4 * np.pi) for _ in range(len(gate_dict_1['theta_indices']))]
        gate_dict_1['theta'] = theta1

        circuit = runner.create_full_circuit(gate_dict, gate_dict_0, gate_dict_1)
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        m0 = int(result.measurements['m0'])
        m1 = int(result.measurements['m1'])
        self.assertIn(m0, [0, 1])
        self.assertIn(m1, [0, 1])

    @staticmethod
    def kron_list(l):
        for i in range(len(l)):
            if i == 0:
                out = l[i]
            else:
                out = np.kron(out, l[i])
        return out

    def test_calculate_probabilities(self):
        runner = CirqRunner(sim_repetitions=100)
        qubits = runner.qubits
        zero = np.array([[1, 0], [0, 0]])
        z_list = [zero for _ in range(4)]
        zero_state = self.kron_list(z_list).astype(np.complex64)
        circuit = cirq.Circuit.from_ops([cirq.X(x) for x in qubits])
        circuit.append(cirq.measure(qubits[0], key='m0'))
        circuit.append(cirq.Circuit.from_ops([cirq.X(x) for x in qubits]))
        circuit.append(cirq.measure(qubits[1], key='m1'))
        probs0 = runner.calculate_probabilities_sampling(zero_state, circuit)
        np.testing.assert_array_equal(probs0, [0, 0, 1, 0])

        one = np.array([[0, 0], [0, 1]])
        o_list = [one for _ in range(4)]
        one_state = self.kron_list(o_list).astype(np.complex64)
        probs1 = runner.calculate_probabilities_sampling(one_state, circuit)
        np.testing.assert_array_equal(probs1, [0, 1, 0, 0])

        circuit2 = cirq.Circuit.from_ops([cirq.H(x) for x in qubits])
        circuit2.append(cirq.measure(qubits[0], key='m0'))
        circuit2.append(cirq.measure(qubits[1], key='m1'))
        probs2 = runner.calculate_probabilities_sampling(zero_state, circuit2)
        np.testing.assert_array_almost_equal(probs2, [0.25, 0.25, 0.25, 0.25], decimal=1)

    def test_compare_prob_methods(self):
        runner = CirqRunner(sim_repetitions=10000)
        dicts = TestLossFromState.get_some_dicts()
        for gate_dict in dicts:
            gate_dict['theta'] = [tf.Variable(x) for x in gate_dict['theta']]
        zero, _ = TestLossFromState.get_some_states()
        zero_ = copy.copy(zero)
        circuit_a = runner.create_full_circuit(dicts[0], dicts[1], dicts[2])
        probs_a = runner.calculate_probabilities_sampling(zero, circuit_a)
        for d in dicts:
            for key, val in d.items():
                if key == 'gate_id':
                    d[key] = np.append(d[key], 4)
                elif key == 'qid':
                    d[key] = np.append(d[key], 0)

        probs_b = runner.calculate_probabilities((dicts[0], dicts[1], dicts[2]), zero_)

        np.testing.assert_almost_equal(np.sum(probs_a), 1)
        np.testing.assert_array_almost_equal(probs_a, probs_b, decimal=4)

