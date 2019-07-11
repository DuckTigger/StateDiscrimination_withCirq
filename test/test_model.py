import tensorflow as tf
import numpy as np
import cirq
import copy

from base_model import Model
from cirq_runner import CirqRunner


class TestLossFromState(tf.test.TestCase):


    @staticmethod
    def kron_list(l):
        for i in range(len(l)):
            if i == 0:
                out = l[i]
            else:
                out = np.kron(out, l[i])
        return out

    @staticmethod
    def get_some_dicts():
        gate_dict = {
            'gate_id': np.array([1, 1, 4, 4]),
            'theta': np.array([np.pi, np.pi]),
            'theta_indices': np.array([0, 1]),
            'control_qid': np.array([]),
            'control_indices': np.array([]),
            'qid': np.array([1, 2, 3, 4])
        }

        gate_dict_0 = {
            'gate_id': np.array([1]),
            'theta': np.array([np.pi]),
            'theta_indices': np.array([0]),
            'control_qid': np.array([]),
            'control_indices': np.array([]),
            'qid': np.array([2])
        }

        gate_dict_1 = {
            'gate_id': np.array([1]),
            'theta': np.array([np.pi]),
            'theta_indices': np.array([0]),
            'control_qid': np.array([]),
            'control_indices': np.array([]),
            'qid': np.array([2])
        }
        return gate_dict, gate_dict_0, gate_dict_1

    @staticmethod
    def get_some_states():
        zero = np.array([[1, 0], [0, 0]])
        one = np.array([[0, 0], [0, 1]])
        z_list = [zero for _ in range(4)]
        oozz = copy.copy(z_list)
        oozz[0] = one
        oozz[1] = one
        zero_state = TestLossFromState.kron_list(z_list).astype(np.complex64)
        oozz_state = TestLossFromState.kron_list(oozz).astype(np.complex64)
        return zero_state, oozz_state

    def test_perfect_discrimination(self):
        """
        If we start in the all 0 state, apply X to all qubits, gate_dict_0 applies XZZ, but should not be reached
        gate_dict_1 applies X again so our measurement outcomes should all be in 10. If we label this state with 0
        this should be complete success.

        If we start in |1000>, all probability should end in 01. This is because now gate_dict_0 is implemented.
        So if we label with 1 we should again get complete
        success.
        """
        zero_state, oozz_state = self.get_some_states()
        zero_state = tf.constant(zero_state)
        oozz_state = tf.constant(oozz_state)
        z_lab = tf.constant(0, dtype=tf.complex64)
        o_lab = tf.constant(1, dtype=tf.complex64)

        gate_dict, gate_dict_0, gate_dict_1 = self.get_some_dicts()

        runner = CirqRunner(sim_repetitions=100)
        loss_calc = Model(1., 1., runner)
        loss_calc.set_all_dicts(gate_dict, gate_dict_0, gate_dict_1)
        states = tf.stack([zero_state, oozz_state])
        labels = tf.stack([z_lab, o_lab])
        loss = []
        for state, label in zip(states, labels):
            prob = loss_calc.state_to_prob(state)
            print(prob)
            loss_i = loss_calc.state_to_loss(state, label)
            loss.append(loss_i)
        loss = [tf.cast(x, tf.float32) for x in loss]
        self.assertAlmostEqual(tf.reduce_mean(loss).numpy(), 0)

    def test_with_circuit(self):
        zero_state, oozz_state = self.get_some_states()
        z_lab = tf.constant(0, dtype=tf.complex64)
        o_lab = tf.constant(1, dtype=tf.complex64)
        labels = tf.stack([z_lab, o_lab])

        runner = CirqRunner(sim_repetitions=100)
        model = Model(1., 1., runner)
        qubits = model.runner.get_qubits()
        circuit = cirq.Circuit.from_ops([cirq.X(x) for x in qubits])
        circuit_0 = cirq.Circuit.from_ops(cirq.X(qubits[1]))
        circuit_1 = cirq.Circuit.from_ops(cirq.X(qubits[1]))
        controlled_0 = model.runner.yield_controlled_circuit(circuit_0, qubits[0])
        controlled_1 = model.runner.yield_controlled_circuit(circuit_1, qubits[0])

        circuit.append(cirq.measure(qubits[0], key='m0'), strategy=cirq.InsertStrategy.NEW)
        circuit.append(controlled_1)
        circuit.append(cirq.X(qubits[0]))
        circuit.append(controlled_0)
        circuit.append(cirq.measure(qubits[1], key='m1'), strategy=cirq.InsertStrategy.NEW)

        states = [zero_state, oozz_state]
        probs = []
        loss = []
        for state, label in zip(states, labels):
            p = model.runner.calculate_probabilities(state, circuit)
            probs.append(p)
            l = model.probs_to_loss(p, label)
            loss.append(l)
        print(probs)
        self.assertAlmostEqual(np.mean(loss), 0)

    def test_variable_setter_getter(self):
        runner = CirqRunner(sim_repetitions=100)
        model = Model(tf.constant(1., dtype=tf.float64), tf.constant(1., dtype=tf.float64), runner)
        gate_dict, gate_dict_0, gate_dict_1 = self.get_some_dicts()
        model.set_all_dicts(gate_dict, gate_dict_0, gate_dict_1)

        gate_dict, gate_dict_0, gate_dict_1 = model.return_gate_dicts()
        n_vars = len(gate_dict['theta_indices']) + len(gate_dict_0['theta_indices']) + len(gate_dict_1['theta_indices'])
        theta = [np.random.rand(1)*4*np.pi for _ in
                 range(n_vars + 100)]
        variables = [tf.Variable(x) for x in theta]
        model.set_variables(variables)
        vars_out = model.get_variables()
        output = [x.numpy() for x in vars_out]
        np.testing.assert_array_almost_equal(output, theta[:n_vars])

