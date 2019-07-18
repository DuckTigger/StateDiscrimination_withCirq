#
# Quantum Machine Learning
# Copyright (C) 2017-2018 Hongxiang Chen, Leonard Wossnig - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Written by :
# - Hongxiang Chen <h.chen.17@ucl.ac.uk>, 2018.
# - Leonard Wossnig, 2017
# - Andrew Patterson, 2018
#

import tensorflow as tf
import numpy as np
import random
import copy


class QSimulator:

    def __init__(self, n, amplitude_damping=0.1):
        """
        Create a *QSimulator* operating on *n* qubits. Visually, the *n* qubits are ordered
        from left to right.

        .. todo::

            1. Use logging instead of *print* for the verbose functionality.
            2. Should we initialise the random number generator? (currently we do)

        Parameters
        ----------
        n : int
            The number of qubits this *QSimulator* operates on.
        verbose : bool
            Open the verbose mode by setting this *True*. On the verbose mode, this *QSimulator* will *print* every
            operation performed. Good for debugging purpose. :math:*\omega*


        """

        self._n = n
        self._amplitude_damping = amplitude_damping
        random.seed()

    @staticmethod
    def kronecker_product(mat1, mat2):
        """Computes the Kronecker product two matrices."""
        m1, n1 = mat1.get_shape()
        mat1_rsh = tf.reshape(mat1, [m1, 1, n1, 1])
        m2, n2 = mat2.get_shape()
        mat2_rsh = tf.reshape(mat2, [1, m2, 1, n2])
        return tf.reshape(mat1_rsh * mat2_rsh, [m1 * m2, n1 * n2])

    @staticmethod
    def convert_vector_to_matrix(wf):
        """

        parameters
            wf: a np array representing a pure state vector
        returns:
            rho: a tensorflow density matrix of the form \bra{wf}\ket{wf}
        """

        ket = copy.copy(wf)
        ket = tf.constant(ket, dtype=tf.complex128)
        bra = tf.transpose(a=ket)
        rho = tf.einsum('i,j->ij', ket, bra)

        return rho

    def matrix_X(self, rho_in, qid):
        """Applies an X gate to the given qubit, now using the density matrix formalism
        parameters:
            rho_in: the input density matrix
            qid: The 'external' qubit ID

        returns:
            rho_out: The density matrix (tf object) once the X transformation has been applied
        """

        x_mat = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)

        gate_mat = self.gate_matrix_1q(qid, x_mat)
        rho_out = self.apply_matrix_to_rho(rho_in, gate_mat)
        return rho_out

    def matrix_Z(self, rho_in, qid):
        """Applies a Z gate to the given qubit, now using the density matrix formalism
                parameters:
                    rho_in: the input density matrix
                    qid: The 'external' qubit ID, i.e. 1st, 2nd, etc. qubit

                returns:
                    rho_out: The density matrix (tf object) once the X transformation has been applied
        """

        z_mat = tf.constant([[1, 0], [0, -1]], dtype=tf.complex128)
        gate_mat = self.gate_matrix_1q(qid, z_mat)
        rho_out = self.apply_matrix_to_rho(rho_in, gate_mat)
        return rho_out

    def matrix_Y(self, rho_in, qid):
        """Applies a Y gate to the given qubit, now using the density matrix formalism
                parameters:
                    rho_in: the input density matrix
                    qid: The 'external' qubit ID, i.e. 1st, 2nd, etc. qubit

                returns:
                    rho_out: The density matrix (tf Tensor) once the X transformation has been applied
        """

        y_mat = tf.constant([[0, -1j], [1j, 0]], dtype=tf.complex128)
        gate_mat = self.gate_matrix_1q(qid, y_mat)
        rho_out = self.apply_matrix_to_rho(rho_in, gate_mat)
        return rho_out

    def Rx(self, rho_in, theta, qid):
        """
        Applies a rotation about the X axis of angle theta to qubit given by qid
        :param rho_in: The density matrix to apply the gate to
        :param qid: The qubit to aply the gate to
        :param theta: The angle to rotate round the x axis by
        :return: rho_out: The resultant density matrix
        """

        theta = tf.cast(tf.divide(theta, 2.), dtype=tf.complex128)
        cos = tf.cos(theta)
        sin = tf.multiply(tf.sin(theta), -1j)
        c1 = tf.stack([cos, sin])
        c2 = tf.stack([sin, cos])
        Rx = tf.stack([c1, c2])
        gate_mat = self.gate_matrix_1q(qid, Rx)
        rho_out = self.apply_matrix_to_rho(rho_in, gate_mat)

        return rho_out

    def Ry(self, rho_in, theta, qid):
        """
        Applies a rotation about the Y axis of angle theta to qubit given by qid
        :param rho_in: The density matrix to apply the gate to
        :param qid: The qubit to apply the gate to
        :param theta: The angle to rotate round the y axis by
        :return: rho_out: The resultant density matrix
        """

        theta = tf.cast(tf.divide(theta, 2.), dtype=tf.complex128)
        cos = tf.cos(theta)
        sin = tf.sin(theta)
        c1 = tf.stack([cos, -sin])
        c2 = tf.stack([sin, cos])
        Ry = tf.stack([c1, c2])
        gate_mat = self.gate_matrix_1q(qid, Ry)
        rho_out = self.apply_matrix_to_rho(rho_in, gate_mat)

        return rho_out

    def Rz(self, rho_in, theta, qid):
        """
        Applies a rotation about the Z axis of angle theta to qubit given by qid
        :param rho_in: The density matrix to apply the gate to
        :param qid: The qubit to apply the gate to
        :param theta: The angle to rotate round the z axis by
        :return: rho_out: The resultant density matrix
        """
        theta = tf.cast(tf.divide(theta, 2.), dtype=tf.complex128)
        e_plus = tf.exp(tf.multiply(theta, 1j))
        e_minus = tf.math.conj(e_plus)
        c1 = tf.stack([e_minus, 0])
        c2 = tf.stack([0, e_plus])
        Rz = tf.stack([c1, c2])
        gate_mat = self.gate_matrix_1q(qid, Rz)
        rho_out = self.apply_matrix_to_rho(rho_in, gate_mat)

        return rho_out

    def Rx_mat(self, theta, qid):
        """
        Applies a rotation about the X axis of angle theta to qubit given by qid
        :param rho_in: The density matrix to apply the gate to
        :param qid: The qubit to apply the gate to
        :param theta: The angle to rotate round the x axis by
        :return: rho_out: The resultant density matrix
        """

        theta = tf.cast(tf.divide(theta, 2.), dtype=tf.complex128)
        cos = tf.cos(theta)
        sin = tf.multiply(tf.sin(theta), -1j)
        c1 = tf.stack([cos, sin])
        c2 = tf.stack([sin, cos])
        Rx = tf.stack([c1, c2])
        gate_mat = self.gate_matrix_1q(qid, Rx)

        return gate_mat

    def Ry_mat(self, theta, qid):
        """
        Applies a rotation about the Y axis of angle theta to qubit given by qid
        :param rho_in: The density matrix to apply the gate to
        :param qid: The qubit to apply the gate to
        :param theta: The angle to rotate round the y axis by
        :return: rho_out: The resultant density matrix
        """

        theta = tf.cast(tf.divide(theta, 2.), dtype=tf.complex128)
        cos = tf.cos(theta)
        sin = tf.sin(theta)
        c1 = tf.stack([cos, -sin])
        c2 = tf.stack([sin, cos])
        Ry = tf.stack([c1, c2])
        gate_mat = self.gate_matrix_1q(qid, Ry)

        return gate_mat

    def Rz_mat(self, theta, qid):
        """
        Applies a rotation about the Z axis of angle theta to qubit given by qid
        :param rho_in: The density matrix to apply the gate to
        :param qid: The qubit to apply the gate to
        :param theta: The angle to rotate round the z axis by
        :return: rho_out: The resultant density matrix
        """
        theta = tf.cast(tf.divide(theta, 2.), dtype=tf.complex128)
        e_plus = tf.exp(tf.multiply(theta, 1j))
        e_minus = tf.math.conj(e_plus)
        c1 = tf.stack([e_minus, 0])
        c2 = tf.stack([0, e_plus])
        Rz = tf.stack([c1, c2])
        gate_mat = self.gate_matrix_1q(qid, Rz)

        return gate_mat

    def hadamard_mat(self, qid):
        """"
        Creates a Hadamard gate for the given qubit
        """
        H = tf.constant([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]], dtype=tf.complex128)
        gate_mat = self.gate_matrix_1q(qid, H)

        return gate_mat

    def gate_matrix_1q(self, qid, gate_matrix):
        """
        Creates the matrix, lifted to the size of the whole Hilbert space,
        for a gate applied to a single qubit.

        Parameters:
        qid: The (external) qubit number that the gate is applied to
        gate_matrix: The 2x2 (single qubit) matrix of the gate


        :return:
        full_matrix: The matrix lifted to the size of the whole Hilbert space
        """
        ops = [tf.eye(2, dtype=tf.complex128) for i in range(self._n)]
        ops[qid] = gate_matrix
        for i in range(len(ops)):
            if i == 0:
                out = ops[i]
            else:
                out = self.kronecker_product(out, ops[i])
        full_matrix = out
        return full_matrix

    def two_qubit_independent_gate(self, qid_0, qid_1, mat_0, mat_1):
        ops = [tf.eye(2, dtype=tf.complex128) for i in range(self._n)]
        ops[qid_0] = mat_0
        ops[qid_1] = mat_1
        for i in range(len(ops)):
            if i == 0:
                out = ops[i]
            else:
                out = self.kronecker_product(out, ops[i])
        full_matrix = out
        return full_matrix

    def control_not_gate_mat(self, qid_control, qid_target):
        n = self._n - 1
        control = qid_control
        target = qid_target
        indices = list(range(2 ** self._n))
        reordered = copy.copy(indices)

        xor_tar = 2 ** (n - target)
        check_con = 2 ** (n - control)
        for i in range(len(indices)):
            if indices[i] & check_con == check_con:
                reordered[i] = indices[i] ^ xor_tar

        iden = tf.eye(2 ** self._n, dtype=tf.complex128)
        cnot = tf.gather(iden, reordered)
        return cnot

    def apply_matrix_to_rho(self, rho_in, full_matrix):
        """
        This applies the (full Hilbert space) gate matrix to the given density matrix.
        It simply applies matrix multiplication

        :param rho_in: The density matrix to apply the gate to
        :param full_matrix: The (full Hilbert space) matrix of the gate (generated by above function)
        :return: rho_out: The matrix once thee gatee has been applied
        """

        rho_out = tf.matmul(full_matrix, tf.matmul(rho_in, full_matrix, adjoint_b=True))

        return rho_out

    def partial_trace(self, rho_in, qid):
        """

        :param rho_in: The density matrix of the multi-patite state in question
        :param qid: The qubit to trace out
        :return: rho_out: The single qubit density matrix, traced out
        """

        no_of_qubits = self._n
        indices = np.array(range(2 * no_of_qubits))
        qubit_index = qid
        second_axis = no_of_qubits + qubit_index
        other_indices = np.delete(indices, [qubit_index, second_axis], axis=0)
        reordering = np.append([qubit_index, second_axis], other_indices)
        shape1 = [2, 2] * no_of_qubits
        shape2 = [2, 2, 2 ** (no_of_qubits - 1), 2 ** (no_of_qubits - 1)]

        reshaped = tf.reshape(rho_in, shape1)
        reordered = tf.transpose(a=reshaped, perm=reordering)
        reshaped2 = tf.reshape(reordered, shape2)

        rho_out = tf.linalg.trace(reshaped2)

        return rho_out

    def measure_rho(self, rho_in, qid):
        """

        :param rho_in:
        :param qid:
        :return: rho_out, the density matrix after measurement
        :return: measured1: bool, whether a 1 was measured on that qubit
        """
        # Initially we will only deal with the computational basis, option to add more bases later
        # measure_one = tf.constant([[0, 0], [0, 1]], dtype=tf.complex128)
        # rho_reduced = self.partial_trace(rho_in, qid)
        # prob_1 = tf.cast(tf.real(tf.trace(tf.matmul(M1, tf.matmul(rho_in, M1, adjoint_b=True)))), dtype=tf.float32)

        one = tf.constant([[0, 0], [0, 1]], dtype=tf.complex128)
        M1 = self.gate_matrix_1q(qid, one)

        prob_1 = tf.linalg.trace(tf.matmul(M1, tf.matmul(M1, rho_in), adjoint_a=True))
        prob_1 = tf.reshape(tf.cast(tf.math.real(prob_1), dtype=tf.float32), [])

        randNum = tf.random.uniform([], 0, 1, dtype=tf.float32)

        def measure1():
            measured1 = tf.constant(1, dtype=tf.complex128)
            return measured1

        def measure0():
            measured1 = tf.constant(0, dtype=tf.complex128)
            return measured1

        measured1 = tf.case([(tf.greater(prob_1, randNum), measure1)], default=measure0)
        return measured1

    def return_probabilites_0_1(self, rho_in, qid):
        """
        Returns an array containing the probabilities of measuring |1> and |0>
        Does not change the density matrix
        :param rho_in: The density matrix to be meaured
        :param qid: The qubit number
        :param full_rho: A flag if the matrix being passed
        :return: probs
        """
        one = tf.constant([[0, 0], [0, 1]], dtype=tf.complex128)
        zero = tf.constant([[1, 0], [0, 0]], dtype=tf.complex128)

        M1 = self.gate_matrix_1q(qid, one)
        M0 = self.gate_matrix_1q(qid, zero)
        prob_1 = tf.linalg.trace(tf.matmul(M1, tf.matmul(M1, rho_in), adjoint_a=True))
        prob_0 = tf.linalg.trace(tf.matmul(M0, tf.matmul(M0, rho_in), adjoint_a=True))

        prob_1 = tf.reshape(tf.cast(tf.math.real(prob_1), dtype=tf.float64), [])
        prob_0 = tf.reshape(tf.cast(tf.math.real(prob_0), dtype=tf.float64), [])

        return [prob_0, prob_1]

    def return_joint_probabilities(self, rho_in, qid_0, qid_1):
        one = tf.constant([[0, 0], [0, 1]], dtype=tf.complex128)
        zero = tf.constant([[1, 0], [0, 0]], dtype=tf.complex128)
        M00 = self.two_qubit_independent_gate(qid_0, qid_1, zero, zero)
        M01 = self.two_qubit_independent_gate(qid_0, qid_1, zero, one)
        M10 = self.two_qubit_independent_gate(qid_0, qid_1, one, zero)
        M11 = self.two_qubit_independent_gate(qid_0, qid_1, one, one)

        p00 = tf.linalg.trace(tf.matmul(M00, rho_in))
        p01 = tf.linalg.trace(tf.matmul(M01, rho_in))
        p10 = tf.linalg.trace(tf.matmul(M10, rho_in))
        p11 = tf.linalg.trace(tf.matmul(M11, rho_in))

        p00 = tf.reshape(tf.cast(tf.math.real(p00), dtype=tf.float64), [])
        p01 = tf.reshape(tf.cast(tf.math.real(p01), dtype=tf.float64), [])
        p10 = tf.reshape(tf.cast(tf.math.real(p10), dtype=tf.float64), [])
        p11 = tf.reshape(tf.cast(tf.math.real(p11), dtype=tf.float64), [])

        return tf.stack([p00, p01, p10, p11])

    def _set_qubit_to_value(self, rho_in, state_to_set: int, qid):
        """

        :param rho_in: Tensor The density matrix to be changed 'measured'
        :param state_to_set: If zero, will set the qubit state to |0>, similarly for |1>, if -1, there was no probability
        of measuring that state, so sets to zeros
        :param qid: the qubit to be changed
        :return:
        """
        measure_zero = tf.constant([[1, 0], [0, 0]], dtype=tf.complex128)
        measure_one = tf.constant([[0, 0], [0, 1]], dtype=tf.complex128)
        null_state = tf.zeros(tf.shape(input=rho_in), dtype=tf.complex128)

        if state_to_set:
            M1 = self.gate_matrix_1q(qid, measure_one)
            rho_out = tf.matmul(M1, tf.matmul(rho_in, M1, adjoint_b=True))
            prob_1 = tf.linalg.trace(tf.matmul(M1, tf.matmul(M1, rho_in), adjoint_a=True))
            rho_out = tf.cond(pred=tf.equal(prob_1, tf.constant(0, dtype=tf.complex128)), true_fn=lambda: null_state,
                              false_fn=lambda: tf.divide(rho_out, prob_1))
        elif state_to_set == 0:
            M0 = self.gate_matrix_1q(qid, measure_zero)
            rho_out = tf.matmul(M0, tf.matmul(rho_in, M0, adjoint_b=True))
            prob_0 = tf.linalg.trace(tf.matmul(M0, tf.matmul(M0, rho_in), adjoint_a=True))
            rho_out = tf.cond(pred=tf.equal(prob_0, tf.constant(0, dtype=tf.complex128)), true_fn=lambda: null_state,
                              false_fn=lambda: tf.divide(rho_out, prob_0))
        else:
            raise ValueError('The value to set the qubit to is incorrect')
        return rho_out

    def apply_kraus_ops(self, rho_in, kraus_ops):
        """
        Takes the input denstiy matrix and applies incoherent noise in the form of Kraus ooperators
        :param rho_in:
        :param kraus_operators: A list of operators (Tensors) to apply to the density matrix, will check if they are valid
        :return: rho_out:
        """
        dims = tf.shape(input=rho_in)
        rho_out = tf.zeros(dims, dtype=tf.complex128)
        for k in kraus_ops:
            r = tf.matmul(k, tf.matmul(rho_in, k, adjoint_b=True))
            rho_out = tf.add(rho_out, r)
        return rho_out

    def amplitude_damping_kops(self, p=None):
        """
        Returns the Kraus operators for a depolarising channel which acts with probability p, on the quibit qid
        :param p:
        :param qid: The qubit to act upon
        :return: A list of Kraus operators which can be inserted into the apply Kraus ops function
        """
        if p is None:
            p = self._amplitude_damping

        k1 = tf.sqrt(tf.constant([[1, 0], [0, (1 - p)]], dtype=tf.complex128))
        k2 = tf.sqrt(tf.constant([[0, p], [0, 0]], dtype=tf.complex128))
        return [k1, k2]

    def two_qubit_kops(self, qid_0, qid_1, kops_1, kops_2):
        k1 = self.two_qubit_independent_gate(qid_0, qid_1, kops_1[0], kops_2[0])
        k2 = self.two_qubit_independent_gate(qid_0, qid_1, kops_1[1], kops_2[1])
        return [k1, k2]

    def single_qubit_kops(self, qid, kops):
        k1 = self.gate_matrix_1q(qid, kops[0])
        k2 = self.gate_matrix_1q(qid, kops[1])
        return [k1, k2]

    def bit_flip_kops(self):
        k1 = tf.sqrt(tf.constant([[0.5, 0], [0, 0.5]], dtype=tf.complex128))
        k2 = tf.sqrt(tf.constant([[0, 0.5], [0.5, 0]], dtype=tf.complex128))
        return [k1, k2]

    def phase_flip_kops(self):
        k1 = tf.sqrt(tf.constant([[0.5, 0], [0, 0.5]], dtype=tf.complex128))
        k2 = tf.sqrt(tf.constant([[0.5, 0], [0, -0.5 + 0 * 1j]], dtype=tf.complex128))
        return [k1, k2]

    def bit_phase_flip_kops(self):
        k1 = tf.sqrt(tf.constant([[0.5, 0], [0, 0.5]], dtype=tf.complex128))
        k2 = tf.sqrt(tf.constant([[0, -0.5j], [0.5j, 0]], dtype=tf.complex128))
        return [k1, k2]

    def depolarising_channel(self):
        k1 = tf.constant([[1, 0], [0, 1]], dtype=tf.complex128)
        k2 = tf.constant([[0, 1], [1, 0]], dtype=tf.complex128)
        k3 = tf.constant([[0, 0 + -1j], [0 + 1j, 0]], dtype=tf.complex128)
        k4 = tf.constant([[1, 0], [0, -1]], dtype=tf.complex128)
        return [k1, k2, k3, k4]

    def depolarising_kops_1q(self, qid):
        kops = self.depolarising_channel()
        kops1q = [self.gate_matrix_1q(qid, k) for k in kops]
        return kops1q

    def deporlarising_kops_2q(self, qid0, qid1):
        kops = self.depolarising_channel()
        kops2q = [self.two_qubit_independent_gate(qid0, qid1, x, y) for x in kops for y in kops]
        return kops2q

    def apply_kops(self, rho_in, noise_prob, kops_in):
        kops = [tf.identity(x) for x in kops_in]
        n_of_ops = len(kops)
        for i in range(n_of_ops):
            if i == 0:
                kops[i] = tf.multiply(
                    tf.cast(np.sqrt(1 - (((n_of_ops - 1) * noise_prob) / n_of_ops)), dtype=tf.complex128), kops[i])
                rho_out = tf.matmul(kops[i], tf.matmul(rho_in, kops[i]))
            else:
                kops[i] = tf.multiply(tf.cast(np.sqrt(noise_prob / n_of_ops), dtype=tf.complex128), kops[i])
                st = tf.matmul(kops[i], tf.matmul(rho_in, kops[i]))
                rho_out = tf.add(rho_out, st)
        return rho_out

    def kops_channel(self, noise_prob, kops_in):
        kops = [tf.identity(x) for x in kops_in]
        n_of_ops = len(kops)
        for i in range(n_of_ops):
            if i == 0:
                kops[i] = tf.multiply(
                    tf.constant(np.sqrt(1 - (((n_of_ops - 1) * noise_prob) / n_of_ops)), dtype=tf.complex128), kops[i])
                channel = tf.matmul(kops[i], kops[i], adjoint_a=True)
            else:
                kops[i] = tf.multiply(tf.constant(np.sqrt(noise_prob / n_of_ops), dtype=tf.complex128), kops[i])
                channel = tf.add(channel, tf.matmul(kops[i], kops[i], adjoint_a=True))
        return channel

    def apply_gate_dict(self, gate_dict, rho_in, noise_on=False, noise_prob=0.1):
        """
        Constructs onr gate matrix to be applied to the states.
        :param gate_dict: Defines the operations to be applied
        :return: A matrix which can be multiplied by states
        """

        def gate_to_fn(gate_dict, index):
            label = gate_dict['gate_id'][index]

            if label == 0:
                control_qid = int(gate_dict['control_qid'][np.where(np.isin(gate_dict['control_indices'], index))])
                target = gate_dict['qid'][index]
                gate_mat = self.control_not_gate_mat(control_qid, target)
                return gate_mat
            if label == 1:
                theta_index = np.array(np.where(np.isin(gate_dict['theta_indices'], index))).reshape([])
                theta = gate_dict['theta'][theta_index]
                target = gate_dict['qid'][index]
                return self.Rx_mat(theta, target)
            if label == 2:
                theta_index = np.array(np.where(np.isin(gate_dict['theta_indices'], index))).reshape([])
                theta = gate_dict['theta'][theta_index]
                target = gate_dict['qid'][index]
                return self.Ry_mat(theta, target)
            if label == 3:
                theta_index = np.array(np.where(np.isin(gate_dict['theta_indices'], index))).reshape([])
                theta = gate_dict['theta'][theta_index]
                target = gate_dict['qid'][index]
                return self.Rz_mat(theta, target)
            if label == 4:
                n = self._n
                return tf.eye(2 ** n, dtype=tf.complex128)
            if label == 5:
                target = gate_dict['qid'][index]
                return self.hadamard_mat(target)

        for i in range(len(gate_dict['qid'])):
            if noise_on:
                if i == 0:
                    rho_out = tf.matmul(rho_in, tf.eye(2 ** self._n, dtype=tf.complex128))
                qid = gate_dict['qid'][i]

                if gate_dict['gate_id'][i] == 0:
                    control_qid = int(gate_dict['control_qid'][np.where(np.isin(gate_dict['control_indices'], i))])
                    kops = self.deporlarising_kops_2q(control_qid, qid)
                    noise_prob_run = noise_prob
                else:
                    kops = self.depolarising_kops_1q(qid)
                    noise_prob_run = (4 * noise_prob) / 5  # See Knill04 for justification

                mat_i = gate_to_fn(gate_dict, i)
                rho_out = tf.matmul(mat_i, tf.matmul(rho_out, mat_i, adjoint_b=True))
                rho_out = self.apply_kops(rho_out, noise_prob_run, kops)
                norm = tf.linalg.trace(rho_out)
                null_state = tf.zeros(tf.shape(input=rho_in), dtype=tf.complex128)
                rho_out = tf.cond(pred=tf.equal(norm, tf.constant(0, dtype=tf.complex128)), true_fn=lambda: null_state,
                                  false_fn=lambda: tf.divide(rho_out, norm))

            else:
                if i == 0:
                    rho_out = tf.matmul(rho_in, tf.eye(2 ** self._n, dtype=tf.complex128))

                mat_i = gate_to_fn(gate_dict, i)
                mat_i = tf.cast(mat_i, dtype=tf.complex128)
                rho_out = tf.matmul(mat_i, tf.matmul(rho_out, mat_i, adjoint_b=True))
                norm = tf.linalg.trace(rho_out)
                null_state = tf.zeros(tf.shape(input=rho_in), dtype=tf.complex128)
                rho_out = tf.cond(pred=tf.equal(norm, tf.constant(0, dtype=tf.complex128)), true_fn=lambda: null_state,
                                  false_fn=lambda: tf.divide(rho_out, norm))

        return rho_out