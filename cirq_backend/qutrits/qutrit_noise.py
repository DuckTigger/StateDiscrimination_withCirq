import cirq
from cirq import protocols, value
from cirq.ops import gate_features
from typing import Sequence, Tuple
import numpy as np
import itertools as it


from qutrits.qutrit_ops import QutritMinusGate, QutritPlusGate, QutritZPlusGate, QutritZMinusGate


def create_noise_matrices():
    trit_plus = protocols.unitary(QutritPlusGate())
    trit_minus = protocols.unitary(QutritZMinusGate())
    trit_zplus = protocols.unitary(QutritZPlusGate())
    trit_zminus = protocols.unitary(QutritZMinusGate())
    eye = np.eye(3)

    x_set = [eye, trit_plus, trit_minus]
    z_set = [eye, trit_zplus, trit_zminus]
    prods = [a @ b for a, b in it.product(x_set, z_set)]
    return prods


def two_q_noise_matrices():
    one_q = create_noise_matrices()
    mats = [np.kron(a, b) for a, b in it.product(one_q, one_q)]
    return mats


@value.value_equality
class SingleQutritDepolarizingChannel(gate_features.SingleQubitGate):

    def __init__(self, p):
        self.unitary_matrices = create_noise_matrices()
        self._p = p
        self._p_i = 1 - 8 * p

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        iden = [(self._p_i, self.unitary_matrices[0])]
        iden.extend([(self.p, m) for m in self.unitary_matrices[1:]])
        return tuple(iden)

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return 'qutrit_depolarize(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'qutrit_depolarize(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> str:
        return '3+D({!r})'.format(self._p)


def qutrit_depolarise(p: float) -> SingleQutritDepolarizingChannel:
    return SingleQutritDepolarizingChannel(p)


@value.value_equality
class TwoQutritDepolarizingChannel(gate_features.TwoQubitGate):

    def __init__(self, p: float):
        self.unitary_matrices = two_q_noise_matrices()
        self._p = p
        self._p_i = 1 - 80 * p

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        iden = [(self._p_i, self.unitary_matrices[0])]
        iden.extend([(self.p, m) for m in self.unitary_matrices[1:]])
        return tuple(iden)

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return 'two_qutrit_depolarize(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'two_qutrit_depolarize(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs) -> str:
        return '2*3+D({!r})'.format(self._p)


def two_qutrit_depolarize(p: float) -> TwoQutritDepolarizingChannel:
    return TwoQutritDepolarizingChannel(p)
