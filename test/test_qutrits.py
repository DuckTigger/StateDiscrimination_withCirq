import unittest
import cirq
import numpy as np

from cirq_backend.qutrits.qutrit_ops import QutritPlusGate, QutritZPlusGate, QutritMinusGate, QutritZMinusGate, C2QutritPlusGate,\
    C2QutritQubitXGate
from cirq_backend.qutrits.qutrit_noise import TwoQutritDepolarizingChannel, SingleQutritDepolarizingChannel,\
    two_qutrit_depolarize, qutrit_depolarise


class TestQutritOps(unittest.TestCase):
    def test_qutrit_circuit(self):
        qutrits = [cirq.LineQid(0, dimension=3), cirq.LineQid(1, dimension=3)]
        circuit = cirq.Circuit(
            QutritZPlusGate().on(qutrits[0]),
            QutritZMinusGate().on(qutrits[0]),
            QutritPlusGate().on(qutrits[1]),
            QutritMinusGate().on(qutrits[1]),
            cirq.measure(*qutrits, key='m')
        )
        sim = cirq.Simulator()
        res = sim.run(circuit, repetitions=10)
        for meas in res.measurements['m']:
            np.testing.assert_array_equal(meas, [0,0])


if __name__ == '__main__':
    unittest.main()
