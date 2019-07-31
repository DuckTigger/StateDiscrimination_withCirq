import cirq
import numpy as np
import itertools as it
from cirq_trainer.noise_model import TwoQubitNoiseModel, two_qubit_depolarize


class TestNoiseModel(np.testing.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qubits = [cirq.NamedQubit(str(x)) for x in range(4)]
        self.noise_prob = 0.001

    @staticmethod
    def kron_list(l):
        for i in range(len(l)):
            if i == 0:
                out = l[i]
            else:
                out = np.kron(out, l[i])
        return out

    @staticmethod
    def return_1q_channel():
        x = np.array([[0, 1], [1, 0]])
        y = np.array([[0, -1j], [1j, 0]])
        z = np.array([[1, 0], [0, -1]])
        i = np.array([[1, 0], [0, 1]])
        return [i, x, y, z]

    def return_2q_channel(self):
        ch_1q = self.return_1q_channel()
        ch_prod = it.product(ch_1q, ch_1q)
        ch_2q = []
        for op in ch_prod:
            ch_2q.append(np.kron(op[0], op[1]))
        return ch_2q

    @staticmethod
    def apply_kops(p, state, kops):
        p_div = len(kops) - 1
        out = []
        for i, op in enumerate(kops):
            if i == 0:
                st = (1-p)*(op @ state @ op.conj().T)
                out.append(st)
            else:
                st = (p/p_div)*(op @ state @ op.conj().T)
                out.append(st)
        out = np.sum(out, axis=0)
        return out

    def noise_1q(self, p, state):
        kops = self.return_1q_channel()
        return self.apply_kops(p, state, kops)

    def channel_1q_multi(self, n_q, target, kop):
        ch = [np.eye(2) for _ in range(n_q)]
        ch[target] = kop
        out = self.kron_list(ch)
        return out

    def noise_1q_multi(self, n_q, target, p, state):
        kops = self.return_1q_channel()
        kops_kron = [self.channel_1q_multi(n_q, target, k) for k in kops]
        return self.apply_kops(p, state, kops_kron)

    def noise_2q(self, p, state):
        kops = self.return_2q_channel()
        return self.apply_kops(p, state, kops)

    def test_1q_circuit(self):
        circuit = cirq.Circuit.from_ops(cirq.X(self.qubits[0]))
        sim = cirq.DensityMatrixSimulator(noise=TwoQubitNoiseModel(cirq.depolarize(4 * self.noise_prob / 5),
                                                                           two_qubit_depolarize(self.noise_prob)))
        state = sim.simulate(circuit, initial_state=0).final_density_matrix
        expected_st = np.array([[0, 0], [0, 1]])
        noisy_st = self.noise_1q(4*self.noise_prob / 5, expected_st)
        np.testing.assert_almost_equal(state, noisy_st)

    def test_2q_circuit(self):
        circuit = cirq.Circuit.from_ops(cirq.X(self.qubits[0]), cirq.CNOT(self.qubits[0], self.qubits[1]))
        sim = cirq.DensityMatrixSimulator(noise=TwoQubitNoiseModel(cirq.depolarize(4 * self.noise_prob / 5),
                                                                           two_qubit_depolarize(self.noise_prob)))
        state = sim.simulate(circuit).final_density_matrix
        z = np.array([[1, 0], [0, 0]])
        o = np.array([[0, 0], [0, 1]])
        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        st_0 = np.kron(o, z)
        noisy_st = self.noise_1q_multi(2, 0, 4 * self.noise_prob / 5, st_0)
        st_1 = cnot @ noisy_st @ cnot.conj().T
        expected = self.noise_2q(self.noise_prob, st_1)
        np.testing.assert_almost_equal(state, expected)

    def test_cirq_noise(self):
        circuit = cirq.Circuit.from_ops(cirq.X(self.qubits[0]))
        sim = cirq.DensityMatrixSimulator(noise=cirq.ConstantQubitNoiseModel(cirq.depolarize(4 * self.noise_prob / 5)))
        state = sim.simulate(circuit, initial_state=0).final_density_matrix
        expected_st = np.array([[0, 0], [0, 1]])
        noisy_st = self.noise_1q(4 * self.noise_prob / 5, expected_st)
        np.testing.assert_almost_equal(state, noisy_st)

    def test_no_noise(self):
        # sanity check
        circuit = cirq.Circuit.from_ops(cirq.X(self.qubits[0]))
        sim = cirq.DensityMatrixSimulator()
        state = sim.simulate(circuit).final_density_matrix
        expected_st = np.array([[0, 0], [0, 1]])
        np.testing.assert_almost_equal(state, expected_st)
