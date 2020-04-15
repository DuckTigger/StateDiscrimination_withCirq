import unittest
import numpy as np
import cirq
from generate_data import CreateDensityMatrices
from cirq.sim.wave_function import validate_normalized_state


class MyTestCase(unittest.TestCase):
    def test_random_states(self):
        states_a, states_b = CreateDensityMatrices.create_random_states(10)
        for a, b in zip(states_a, states_b):
            validate_normalized_state(a, qid_shape=(2,))
            validate_normalized_state(b, qid_shape=(2,))

if __name__ == '__main__':
    unittest.main()
