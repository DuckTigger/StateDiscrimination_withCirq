import cirq
import numpy as np
import sympy as sp
from cirq.ops import gate_features, eigen_gate
from sympy.physics.quantum import TensorProduct


class QutritPlusGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    # def _unitary_(self):
    #     return np.array([[0, 0, 1],
    #                      [1, 0, 0],
    #                      [0, 1, 0]])

    def _eigen_components(self):
        rt_3 = np.sqrt(3)
        return [(0,
                 np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])),
                (-2 / 3,
                 np.array([[1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2)),
                            -1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2)),
                            -1 / (1 / 2 + rt_3 * 1j / 2)],
                           [-1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2) ** 2),
                            1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2) ** 2),
                            (1 / 2 + rt_3 * 1j / 2) ** (-2)],
                           [-1 / (1 / 2 - rt_3 * 1j / 2), (1 / 2 - rt_3 * 1j / 2) ** (-2), 1]])),
                (2 / 3,
                 np.array([[1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2)),
                            -1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2) ** 2),
                            -1 / (1 / 2 - rt_3 * 1j / 2)],
                           [-1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2)),
                            1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2) ** 2),
                            (1 / 2 - rt_3 * 1j / 2) ** (-2)],
                           [-1 / (1 / 2 + rt_3 * 1j / 2), (1 / 2 + rt_3 * 1j / 2) ** (-2), 1]]))]

    def _circuit_diagram_info_(self, args):
        return '[+1]'


class QutritMinusGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    # def _unitary_(self):
    #     return np.array([[0, 1, 0],
    #                      [0, 0, 1],
    #                      [1, 0, 0]])

    def _eigen_components(self):
        rt_3 = np.sqrt(3)
        return [(0,
                 np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])),
                (-2 / 3,
                 np.array([[1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2) ** 2),
                            -1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2) ** 2),
                            (1 / 2 + rt_3 * 1j / 2) ** (-2)],
                           [-1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2)),
                            1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2)), -1 / (1 / 2 + rt_3 * 1j / 2)],
                           [(1 / 2 - rt_3 * 1j / 2) ** (-2), -1 / (1 / 2 - rt_3 * 1j / 2), 1]])),
                (2 / 3,
                 np.array([[1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2) ** 2),
                            -1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2)),
                            (1 / 2 - rt_3 * 1j / 2) ** (-2)],
                           [-1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2) ** 2),
                            1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2)), -1 / (1 / 2 - rt_3 * 1j / 2)],
                           [(1 / 2 + rt_3 * 1j / 2) ** (-2), -1 / (1 / 2 + rt_3 * 1j / 2), 1]]))]

    def _circuit_diagram_info_(self, args):
        return '[-1]'


class QutritZPlusGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    # def _unitary_(self):
    #     return np.array([[1, 0, 0],
    #                     [0, np.exp(2j * np.pi / 3), 0],
    #                     [0, 0, np.exp(4j * np.pi / 3)]])

    def _eigen_components(self):
        return [(0,
                 np.array([[1.0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])),
                ((1.0471975511966 - np.pi) / np.pi,
                 np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1.0]])),
                ((-1.0471975511966 + np.pi) / np.pi,
                 np.array([[0, 0, 0],
                           [0, 1.0, 0],
                           [0, 0, 0]]))]

    def _circuit_diagram_info_(self, args):
        return '[+Z]'


class QutritZMinusGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    def _qid_shape_(self):
        return (3,)

    # def _unitary_(self):
    #     return np.array([[1, 0, 0],
    #                      [0, np.exp(4j * np.pi / 3), 0],
    #                      [0, 0, np.exp(2j * np.pi / 3)]])

    def _eigen_components(self):
        return [(0,
                 np.array([
                     [1.0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])),
                ((1.0471975511966 - np.pi) / np.pi,
                 np.array([
                     [0, 0, 0],
                     [0, 1.0, 0],
                     [0, 0, 0]])),
                ((-1.0471975511966 + np.pi) / np.pi,
                 np.array([
                     [0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 1.0]]))]

    def _circuit_diagram_info_(self, args):
        return '[-Z]'


class C2QutritPlusGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """
    2-controlled Qutrit Plus Gate on two qutrits
    """

    def _qid_shape_(self):
        return (3, 3,)

    # def _unitary_(self):
    #     return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                      [0, 0, 0, 0, 0, 0, 0, 1, 0]])

    def _eigen_components(self):
        rt_3 = np.sqrt(3)
        return [(0,
                 np.array([
                     [1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 1, 1, 1]])),
                (-2 / 3, np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2)),
                     -1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2)), -1 / (1 / 2 + rt_3 * 1j / 2)],
                    [0, 0, 0, 0, 0, 0, -1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2) ** 2),
                     1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2) ** 2),
                     (1 / 2 + rt_3 * 1j / 2) ** (-2)],
                    [0, 0, 0, 0, 0, 0, -1 / (1 / 2 - rt_3 * 1j / 2), (1 / 2 - rt_3 * 1j / 2) ** (-2), 1]])),
                (2 / 3, np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2)),
                     -1 / ((1 / 2 - rt_3 * 1j / 2) * (1 / 2 + rt_3 * 1j / 2) ** 2), -1 / (1 / 2 - rt_3 * 1j / 2)],
                    [0, 0, 0, 0, 0, 0, -1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2)),
                     1 / ((1 / 2 - rt_3 * 1j / 2) ** 2 * (1 / 2 + rt_3 * 1j / 2) ** 2),
                     (1 / 2 - rt_3 * 1j / 2) ** (-2)],
                    [0, 0, 0, 0, 0, 0, -1 / (1 / 2 + rt_3 * 1j / 2), (1 / 2 + rt_3 * 1j / 2) ** (-2), 1]]))]

    def _circuit_diagram_info_(self, args):
        return cirq.protocols.CircuitDiagram1jnfo(
            wire_symbols=('2', '[+1]'))


class C2QutritQubitXGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """
    2-controlled Qutrit-Qubit CNOT gate
    """

    def _qid_shape_(self):
        return (3, 2,)

    # def _unitary_(self):
    #     return np.array([[1, 0, 0, 0, 0, 0],
    #                      [0, 1, 0, 0, 0, 0],
    #                      [0, 0, 1, 0, 0, 0],
    #                      [0, 0, 0, 1, 0, 0],
    #                      [0, 0, 0, 0, 0, 1],
    #                      [0, 0, 0, 0, 1, 0]])

    def _eigen_components(self):
        return [(1,
                 np.array([
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, -1],
                     [0, 0, 0, 0, -1, 1]])),
                (0,
                 np.array([
                     [1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1, 1]]))]

    def _circuit_diagram_info_(self, args):
        return cirq.protocols.CircuitDiagramInfo(
            wire_symbols=('2', '[X]'))


def construct_eigen_space_sp(mat: 'np.ndarray'):
    mat = sp.Matrix(mat)
    eigvecs = mat.eigenvects()
    tuples = []
    for vecs in eigvecs:
        proj = sp.zeros(*mat.shape)
        for vec in vecs[2]:
            proj += TensorProduct(vec, vec.T.conjugate())
        real, imag = vecs[0].as_real_imag()
        phase = sp.atan2(imag, real) / sp.pi
        tuples.append((phase, proj))
    return tuples


if __name__ == '__main__':
    # m = construct_eigen_space_sp(C2QutritQubitXGate()._unitary_())
    # print(sp.simplify(m))
    q0 = cirq.LineQid(0, dimension=3)
    q1 = cirq.LineQid(1, dimension=2)
    qubits = [q0, q1]
    circuit = cirq.Circuit(
        QutritZPlusGate(exponent=0.75).on(q0),
        QutritPlusGate().on(q0),
        C2QutritQubitXGate(exponent=0.5).on(q0, q1),
        # cirq.measure(*qubits, key='result')
    )
    print(circuit)

    # circuit.append(cirq.measure(*q0, 'm'))
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    print(result)
