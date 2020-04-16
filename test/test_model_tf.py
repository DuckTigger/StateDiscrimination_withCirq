import tensorflow as tf
import numpy as np
import copy

from tensorflow_backend.base_model_tf import ModelTF
from tensorflow_backend.tf2_simulator_runner import TF2SimulatorRunner
from shared.gate_dictionaries import GateDictionaries


class TestLossFromStateTF(tf.test.TestCase):

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
            'qid': np.array([0, 1, 2, 3])
        }

        gate_dict_0 = {
            'gate_id': np.array([1, 4, 4]),
            'theta': np.array([np.pi]),
            'theta_indices': np.array([0]),
            'control_qid': np.array([]),
            'control_indices': np.array([]),
            'qid': np.array([1, 2, 3])
        }

        gate_dict_1 = {
            'gate_id': np.array([1, 4, 4]),
            'theta': np.array([np.pi]),
            'theta_indices': np.array([0]),
            'control_qid': np.array([]),
            'control_indices': np.array([]),
            'qid': np.array([1, 2, 3])
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
        zero_state = TestLossFromStateTF.kron_list(z_list).astype(np.complex64)
        oozz_state = TestLossFromStateTF.kron_list(oozz).astype(np.complex64)
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

        runner = TF2SimulatorRunner()
        loss_calc = ModelTF(1., 1., runner)
        loss_calc.set_all_dicts(gate_dict, gate_dict_0, gate_dict_1)
        states = tf.stack([zero_state, oozz_state])
        labels = tf.stack([z_lab, o_lab])
        loss = []
        for state, label in zip(states, labels):
            loss_i = loss_calc.loss_fn(state, label)
            loss.append(loss_i)
        loss = [tf.cast(x, tf.float32) for x in loss]
        self.assertAlmostEqual(tf.reduce_mean(loss).numpy(), 0, places=0)

    def test_variable_setter_getter(self):
        runner = TF2SimulatorRunner()
        model = ModelTF(tf.constant(1., dtype=tf.float64), tf.constant(1., dtype=tf.float64), runner)
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

    def test_gradients(self):
        runner = TF2SimulatorRunner()
        model = ModelTF(tf.constant(1., dtype=tf.float64), tf.constant(1., dtype=tf.float64), runner)
        gate_dict, gate_dict_0, gate_dict_1 = GateDictionaries().return_short_dicts_ran_vars()
        model.set_all_dicts(gate_dict, gate_dict_0, gate_dict_1)
        print('Vars before: {}'.format(model.get_variables()))
        zero_state, oozz_state = self.get_some_states()
        grads = model.variables_gradient_exact(state=tf.constant(zero_state, dtype=tf.complex64),
                                               label=(tf.constant(0, dtype=tf.float32)))
        print('Grads: {}\n Vars:{}'.format(grads, model.get_variables()))

    def test_vars_ids(self):
        runner = TF2SimulatorRunner()
        model = ModelTF(tf.constant(1., dtype=tf.float64), tf.constant(1., dtype=tf.float64), runner)
        gate_dict, gate_dict_0, gate_dict_1 = GateDictionaries.return_new_dicts_rand_vars()
        model.set_all_dicts(gate_dict, gate_dict_0, gate_dict_1)
        ids = model.get_gate_ids()
        var = model.get_variables()
        np.testing.assert_equal(len(ids), len(var))

