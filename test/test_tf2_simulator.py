"""
    Quantum Machine Learning
    Copyright (C) 2018 Hongxiang Chen,Andrew Patterson - All Rights Reserved.
    Unauthorized copying of this file, via any medium is strictly prohibited.
    Written by Hongxiang Chen <h.chen.17@ucl.ac.uk>, Andrew Patterson, <a.patterson.10@ucl.ac.uk> 2018.

"""

# Tests the gates with noise module, refactoring from the gates test
# the output should be checked against the correct outcome.
# This file is so named so that python unittest will not automatically run it.


from tf2_simulator import QSimulator
import numpy as np
import os
import itertools as iter
import tensorflow as tf


class TestNoisyGates(tf.test.TestCase):

    def test_convert_vector_to_matrix(self):
        tester = QSimulator(1)

        a = 0.5
        b = -0.5
        ket = np.array([a, b])
        bra = np.conj(ket)
        
        rho = tester.convert_vector_to_matrix(ket)
        expected = tf.constant(np.einsum('i,j->ij', ket, bra), dtype=tf.complex128)


        self.assertAllEqual(rho, expected)

    def test_XZY_gates(self):
        tester = QSimulator(1)
        a = 0.5
        b = -0.5
        ket = np.array([a, b])
        bra = np.conj(ket)
        rho = np.einsum('i,j->ij', ket, bra)

        # Testing the X gate
    
        rho_in = tf.constant(rho, dtype=tf.complex128)
        rho_out = tester.matrix_X(rho_in, 0)
        ket_out = np.array([b, a])
        bra_out = np.conj(ket_out)
        expected = tf.constant(np.einsum('i,j->ij', ket_out, bra_out), dtype=tf.complex128)


        self.assertAllEqual(rho_out, expected)

        # Testing the Y gate
        
        rho_in = tf.constant(rho, dtype=tf.complex128)
        rho_out = tester.matrix_Y(rho_in, 0)
        y_mat = np.array([[0j, -1j], [1j, 0j]])
        expected = tf.constant(np.einsum('ij,jk->ik', y_mat, np.einsum('ij,jk->ik', rho, y_mat)),
                               dtype=tf.complex128)


        self.assertAllEqual(rho_out, expected)

        # Testing the Z gate
        
        rho_in = tf.constant(rho, dtype=tf.complex128)
        rho_out = tester.matrix_Z(rho_in, 0)
        ket_out = np.array([a, -b])
        bra_out = np.conj(ket_out)
        expected = tf.constant(np.einsum('i,j->ij', ket_out, bra_out), dtype=tf.complex128)


        self.assertAllEqual(rho_out, expected)

    def test_rotation_XYZ_gates(self):
        tester = QSimulator(1)
        a = 0.5
        b = -0.5
        ket = np.array([a, b])
        bra = np.conj(ket)
        rho = np.einsum('i,j->ij', ket, bra)

        
        rho_in = tf.constant(rho, dtype=tf.complex128)
        theta = 0.5
        output = tester.Rx(rho_in, theta, 0)
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        gate = np.array([[cos,-1j*sin],[-1j*sin, cos]])
        expected = np.matmul(gate, np.matmul(rho, np.conj(gate.T)))


        np.testing.assert_almost_equal(output.numpy(), expected)

        
        rho_in = tf.constant(rho, dtype=tf.complex128)
        theta = 0.5
        output = tester.Ry(rho_in, theta, 0)
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        gate = np.array([[cos, -sin], [sin, cos]])
        expected = np.matmul(gate, np.matmul(rho, np.conj(gate.T)))


        np.testing.assert_almost_equal(output.numpy(), expected)

    
        rho_in = tf.constant(rho, dtype=tf.complex128)
        theta = 0.5
        output = tester.Rz(rho_in, theta, 0)
        e_plus = np.exp(1j * (theta / 2))
        e_minus = np.conj(e_plus)
        gate = np.array([[e_minus, 0], [0, e_plus]])
        expected = np.matmul(gate, np.matmul(rho, np.conj(gate.T)))


        np.testing.assert_almost_equal(output.numpy(), expected)

    def test_single_qubit_gate(self):
        
        tester = QSimulator(4)
        # Use an X gate for simplicity
        gate = np.array([[0, 1], [1, 0]])
        gate_tf = tf.constant(gate, dtype=tf.complex128)
        identity = np.eye(2)

        expected = tf.constant(np.kron(identity, np.kron(identity, np.kron(gate, identity))), dtype=tf.complex128)
        gate_out = tester.gate_matrix_1q(2, gate_tf)


        self.assertAllEqual(expected, gate_out)

    def test_control_gate(self):
        # test the case in the apply gatelist
        
        tester = QSimulator(2)
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        mixed = np.array([[0.5, 0.5], [0.5, 0.5]])
        state = np.kron(mixed, one)
        rho_in = tf.constant(state, dtype=tf.complex128)
        # expected = tf.constant(np.kron(one, zero),dtype=tf.complex128)
        expected = tf.constant([[0, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 0]],
                               dtype=tf.complex128)
        output = tester.apply_matrix_to_rho(rho_in, tester.control_not_gate_mat(0, 1))


        self.assertAllEqual(expected, output, msg='CNOT on 1,1')

        
        tester = QSimulator(2)
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        state = np.kron(zero, one)
        rho_in = tf.constant(state, dtype=tf.complex128)
        expected = tf.constant(np.kron(zero, one), dtype=tf.complex128)
        output = tester.apply_matrix_to_rho(rho_in, tester.control_not_gate_mat(0, 1))


        self.assertAllEqual(expected, output, msg='CNOT on 0,1')

    def test_apply_matrix(self):
        
        tester = QSimulator(4)
        # Use the same gate as above, CNOT: c=2, t=4

        gate = np.array([[0, 1], [1, 0]])
        gate_tf = tf.constant(gate, dtype=tf.complex128)
        iden = np.eye(2)
        gate_mat = np.kron(iden, np.kron(iden, np.kron(iden, gate)))

        zero = np.array([[1, 0], [0, 0]])
        one = np.array([[0, 0], [0, 1]])
        # Initial state is |0>|1>|0>|1>
        initial = np.kron(zero, np.kron(one, np.kron(zero, one)))
        # Final state is then |0>|1>|0>|0>
        final = np.kron(zero, np.kron(one, np.kron(zero, zero)))
        expected = tf.constant(final)

        rho_in = tf.constant(initial, dtype=tf.complex128)
        # Testing using simulator fn.
        gate_mat = tester.control_not_gate_mat(1, 3)
        output = tester.apply_matrix_to_rho(rho_in, gate_mat)


        self.assertAllEqual(expected, output)

    def test_partial_trace(self):
    
        tester = QSimulator(6)
        # We will create a generic density matrix with known values at qubit 3
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        a = 0.5
        b = -0.5
        ket = np.array([a, b])
        psi = np.einsum('i,j->ij', ket, np.conj(ket))
        # Create |0>|0>|psi>|1>|1>|1>
        state = np.kron(zero, np.kron(zero, np.kron(psi, np.kron(one, np.kron(one, one)))))

        rho_in = tf.constant(state, dtype=tf.complex128)
        expected = tf.constant(psi, dtype=tf.complex128)
        output = tester.partial_trace(rho_in, 2)


        self.assertAllEqual(expected, output)

    def test_measure_qubit(self):
    
        tester = QSimulator(6)
        # Create a deteministic state and measure the third qubit
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        # |0>|0>|1>|1>|1>|1>
        state = np.kron(zero, np.kron(zero, np.kron(one, np.kron(one, np.kron(one, one)))))

        rho_in = tf.constant(state, dtype=tf.complex128)
        measurement = tester.measure_rho(rho_in, 2)

        measured1 = measurement
        self.assertTrue(measured1.numpy())

    def test_return_probabilities(self):
        tester = QSimulator(1)
        a = 0.25
        b = -0.75
        ket = np.array([a, b])
        bra = np.conj(ket)
        rho = np.einsum('i,j->ij', ket, bra)

    
        rho_in = tf.constant(rho, dtype=tf.complex128)
        theta = 0.5
        apply_gate = tester.Rx(rho_in, theta, 0)
        output = tester.return_probabilites_0_1(apply_gate, 0)
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        gate = np.array([[cos, -1j * sin], [-1j * sin, cos]])
        expected_mat = np.matmul(gate, np.matmul(rho, np.conj(gate.T)))
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        p0 = np.trace(np.matmul(zero, np.matmul(expected_mat, np.conj(zero.T))))
        p1 = np.trace(np.matmul(one, np.matmul(expected_mat, np.conj(one.T))))
        expected = tf.stack([tf.constant(p0, dtype=tf.complex128),tf.constant(p1, dtype=tf.complex128)])
        np.testing.assert_almost_equal(expected.numpy(), output)

    def test_amp_damping_kops(self):
        
        tester = QSimulator(6)
        k1 = np.array([[1, 0], [0, np.sqrt(0.9)]])
        k2 = np.array([[0, np.sqrt(0.1)], [0, 0]])
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        # create |0>|0>|1>|0>|0>|0>
        state = np.kron(zero, np.kron(zero, np.kron(one, np.kron(zero, np.kron(zero, zero)))))
        gate1 = np.kron(np.eye(2 ** 2), np.kron(k1, np.eye(2 ** 3)))
        gate2 = np.kron(np.eye(2 ** 2), np.kron(k2, np.eye(2 ** 3)))
        np_out = np.matmul(gate1, np.matmul(state, np.conj(gate1.T))) + \
                 np.matmul(gate2, np.matmul(state, np.conj(gate2.T)))
        expected = tf.constant(np_out, dtype=tf.complex128)

        rho_in = tf.constant(state, dtype=tf.complex128)
        kops = tester.amplitude_damping_kops(0.1)
        kops = [tester.gate_matrix_1q(2, k) for k in kops]
        output = tester.apply_kraus_ops(rho_in, kops)


        self.assertAllEqual(expected, output)

    
        tester = QSimulator(1)
        k1 = np.array([[1, 0], [0, np.sqrt(0.9)]])
        k2 = np.array([[0, np.sqrt(0.1)], [0, 0]])
        one = np.array([[0, 0], [0, 1]])

        expected = np.matmul(k1, np.matmul(one, k1.T)) + np.matmul(k2, np.matmul(one, k2.T))
        expected = tf.constant(expected, dtype=tf.complex128)
        rho_in = tf.constant(one, dtype=tf.complex128)
        kops = tester.amplitude_damping_kops(0.1)
        output = tester.apply_kraus_ops(rho_in, kops)


        print('np:\n{}\ntf:\n{}\n'.format(expected, output))
        self.assertAllEqual(expected, output)

    def test_apply_gatedict_to_many_rhos_noisy(self):
        
        tester = QSimulator(2)
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        mixed = np.array([[0.5, 0.5], [0.5, 0.5]])

        # Create the state |0>|1>
        state1 = np.kron(zero, one)
        # Create the state |1>|0>
        state2 = np.kron(one, zero)
        state3 = np.kron(mixed, one)

        # Apply the gatelist corresponding to an X on qubit 1, then a CNOT between 1 and 2

        gate_dict = {
            'gate_id': np.array([1, 0]),
            'theta': tf.Variable([np.pi]),
            'theta_indices': np.array([0]),
            'control_qid': np.array([0]),
            'control_indices': np.array([1]),
            'qid': np.array([0, 1])
        }
        expected1 = np.kron(one, zero)
        expected2 = np.kron(zero, zero)
        expected3 = np.array([[0, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 0]])

        rho_list = tf.stack([tf.constant(state1, dtype=tf.complex128), tf.constant(state2, dtype=tf.complex128),
                             tf.constant(state3, dtype=tf.complex128)], axis=0)
        expected = tf.stack(
            [tf.constant(expected1, dtype=tf.complex128), tf.constant(expected2, dtype=tf.complex128),
             tf.constant(expected3, dtype=tf.complex128)], axis=0)

        

        # Turn off noise as it cannot be determined deterministically
        output = tf.map_fn(lambda x:tester.apply_gate_dict(gate_dict, x, noise_on=True, noise_prob=0.0), rho_list)

        self.assertAllClose(expected, output)

    def test_apply_gatedict_noise_on(self):
    
        tester = QSimulator(2)
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        mixed = np.array([[0.5, 0.5], [0.5, 0.5]])

        # Create the state |0>|1>
        state1 = np.kron(zero, one)
        # Create the state |1>|0>
        state2 = np.kron(one, zero)
        state3 = np.kron(mixed, one)
        states = [state1, state2, state3]
        # Apply the gatelist corresponding to an X on qubit 1, then a CNOT between 1 and 2

        gate_dict = {
            'gate_id': np.array([1, 0]),
            'theta': tf.Variable([np.pi]),
            'theta_indices': np.array([0]),
            'control_qid': np.array([0]),
            'control_indices': np.array([1]),
            'qid': np.array([0, 1])
        }

        k1 = np.array([[1, 0], [0, 1]])
        k2 = np.array([[0, 1], [1, 0]])
        k3 = np.array([[0, 0 + -1j], [0 + 1j, 0]])
        k4 = np.array([[1, 0], [0, -1]])
        kops = [k1, k2, k3, k4]

        noise_prob = 0.5
        expected_nonoise1 = np.kron(one, zero)
        expected_nonoise2 = np.kron(zero, zero)
        expected_nonoise3 = np.array([[0, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 0]])
        expected_nonoise = np.stack([expected_nonoise1, expected_nonoise2, expected_nonoise3])

        def gate_mat_1q(qid, mat):
            ops = [np.eye(2) for i in range(2)]
            ops[qid] = mat
            for i in range(len(ops)):
                if i == 0:
                    out = ops[i]
                else:
                    out = np.kron(out, ops[i])
            return out

        def gate_mat_2q(qid_0, qid_1, mat_0, mat_1):
            ops = [np.eye(2) for i in range(2)]
            ops[qid_0] = mat_0
            ops[qid_1] = mat_1
            for i in range(len(ops)):
                if i == 0:
                    out = ops[i]
                else:
                    out = np.kron(out, ops[i])
            return out

        def kops_1q(qid):
            return [gate_mat_1q(qid, k) for k in kops]

        def kops_2q(qid0, qid1):
            return [gate_mat_2q(qid0, qid1, x, y) for x in kops for y in kops]

        def apply_kops(rho_in, kops, noise_prob):
            n_of_ops = len(kops)
            for i in range(n_of_ops):
                if i == 0:
                    kops[i] = (np.sqrt(1 - (((n_of_ops - 1) * noise_prob) / n_of_ops)))*kops[i]
                else:
                    kops[i] = (np.sqrt(noise_prob / n_of_ops)) * kops[i]
            channel = [np.matmul(k, np.matmul(rho_in, k.conj().T)) for k in kops]
            rho_out = np.sum(channel, axis=0)
            return rho_out

        x_1 = gate_mat_1q(0, [[0,1], [1,0]])
        cnot = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]])

        def action(state):
            out = x_1 @ state @ x_1.T
            out = apply_kops(out, kops_1q(0), (4*noise_prob)/5)
            out = cnot @ out @ cnot.T
            out = apply_kops(out, kops_2q(0, 1), noise_prob)
            return out

        expected = [action(state) for state in states]
        expected = [state / np.trace(state) for state in expected]
        expected = np.stack(expected)

        rho_list = tf.stack([tf.constant(state1, dtype=tf.complex128), tf.constant(state2, dtype=tf.complex128),
                            tf.constant(state3, dtype=tf.complex128)], axis=0)

        

        output = tf.map_fn(lambda x: tester.apply_gate_dict(gate_dict, x, noise_on=True, noise_prob=noise_prob), rho_list)


        output_nonoise = tf.map_fn(lambda x: tester.apply_gate_dict(gate_dict, x, noise_on=True, noise_prob=0), rho_list)


        self.assertAllClose(output_nonoise, expected_nonoise)
        self.assertAllClose(output, expected)

    def test_kops_1q(self):
    
        tester = QSimulator(4)
        kops = tester.depolarising_channel()
        for i in range(4):
            kops_1q = [tester.gate_matrix_1q(i, x) for x in kops]
            iden = tf.eye(2**4, dtype=tf.complex128)

            for p in range(11):
                p = p / 10
                channel_1q = tester.kops_channel(p, kops_1q)
                expected = iden
                out_1q = channel_1q

                print('p = {}\n1q channel:\n{}\n'.format(p, np.array(out_1q).real))
                self.assertAllClose(out_1q, expected)

    def test_kops_2q(self):
    
        tester = QSimulator(4)
        kops = tester.depolarising_channel()
        for i, j in iter.permutations([0,1,2,3], 2):
            kops_2q = [tester.two_qubit_independent_gate(i, j, x, y) for x in kops for y in kops]
            iden = tf.eye(2**4, dtype=tf.complex128)

            for p in range(11):
                p = p / 10
                channel_2q = tester.kops_channel(p, kops_2q)
                expected = iden
                out_2q = channel_2q


                print('p = {}\n2q Channel:\n{}'.format(p, np.array(out_2q).real))
                self.assertAllClose(out_2q, expected)

    def test_similar_from_kops_and_noise_off(self):
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        mixed = np.array([[0.5, 0.5], [0.5, 0.5]])

        # Create the state |0>|1>|1>|1>
        state1 = np.kron(zero, np.kron(one, np.kron(one, one)))
        # Create the state |1>|0>|1>|1>
        state2 = np.kron(one, np.kron(zero, np.kron(one, one)))
        state3 = np.kron(mixed, np.kron(mixed, np.kron(mixed, mixed)))
        states = [state1, state2, state3]
        # Apply the gatelist corresponding to an X on qubit 1, then a CNOT between 1 and 2

        gate_dict = {
            'gate_id': np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0]),
            'theta': None,
            'theta_indices': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            'control_qid': np.array([0, 1, 2, 3]),
            'control_indices': np.array([12, 13, 14, 15]),
            'qid': np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0])

        }
        model_loc = 'C:\\Users\\Andrew Patterson\\Documents\\MRes\\tf_logs\\myriad_data\\vary_noise_new_model_save\\2019_05_31_00_25_02'
        angles = np.load(os.path.join(model_loc, 'parameters.npy'))
        gate_dict['theta'] = [tf.Variable(angles[x], dtype=tf.float64, name='theta' + str(x)) for x in
                       range(12)]

        # expected_nonoise1 = np.kron(one, zero)
        # expected_nonoise2 = np.kron(zero, zero)
        # expected_nonoise3 = np.array([[0, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 0]])
        # expected_nonoise = np.stack([expected_nonoise1, expected_nonoise2, expected_nonoise3])

        rho_list = tf.stack([tf.constant(state, dtype=tf.complex128)for state in states], axis=0)

    
        tester = QSimulator(4)
        


        output_nonoise = tf.map_fn(lambda x: tester.apply_gate_dict(gate_dict, x, noise_on=False),
                                   rho_list)

        output_np_0 = tf.map_fn(lambda x: tester.apply_gate_dict(gate_dict, x, noise_on=True, noise_prob=0),
                           rho_list)


        print('Noise off:\n{}\nNoise prob=0:\n{}\n'.format(output_nonoise, output_np_0))
        self.assertAllClose(output_nonoise, output_np_0, atol=0.1)

    def test_summation(self):
        one = np.array([[0, 0], [0, 1]])
        zero = np.array([[1, 0], [0, 0]])
        mixed = np.array([[0.5, 0.5], [0.5, 0.5]])

        # Create the state |0>|1>|1>|1>
        state1 = np.kron(zero, np.kron(one, np.kron(one, one)))
        # Create the state |1>|0>|1>|1>
        state2 = np.kron(one, np.kron(zero, np.kron(one, one)))
        state3 = np.kron(mixed, np.kron(mixed, np.kron(mixed, mixed)))
        states = [state1, state2, state3]

        expected = np.sum(states, axis=0)

    
        states_in = [tf.constant(x, dtype=tf.complex128) for x in states]
        output = tf.reduce_sum(states_in, axis=0)

        print('tf output:\n{}\nnp expected:\n{}\n'.format(output, expected))
        self.assertAllClose(output,expected, atol=1e-3)