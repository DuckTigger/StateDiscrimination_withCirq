import numpy as np
from scipy.stats import truncnorm
import os
import json


class CreateDensityMatrices:
    """
    Mostly ported as is from old scheme. Data generated should then be sound.
    """

    def __init__(self):
        pass
    
    @staticmethod
    def state_from_vec(vec: np.ndarray):
        """
        Takes a two qubit vector, turns it into a density matrix and tensor products it with two qubits
        in the zero state. Note that the state qubits appear in the least significant bit, as the first two qubits are
        those manipulated and measured.
        :param vec: A two qubit state vector
        :return: state: The four qubit density matrix
        """
        zero = np.array([[1, 0], [0, 0]])
        zero_zero = np.kron(zero, zero)
        rho = np.einsum('i,j->ij', vec, np.conj(vec))
        rho = rho / np.trace(rho)
        state = np.kron(zero_zero, rho)
        state = state / np.trace(state)
        return state
        
    @staticmethod
    def create_a(dist_choice, a_const=False):
        """
        Creates constant a states or randmoised ones, and appends it to the state list, out
        :param out: The output state list
        :param dist_choice: the random number from the distribution
        :param a_const:
        :return:
        """
        if a_const:
            psi_a = np.array([np.sqrt(1 - (0.25 ** 2)), 0, 0.25, 0])
        else:
            rand_a = dist_choice
            psi_a = np.array([np.sqrt(1 - (rand_a ** 2)), 0, rand_a, 0])
        state = CreateDensityMatrices.state_from_vec(psi_a)

        assert np.allclose(state, np.conj(state.T))
        assert np.allclose(np.trace(state), 1)
        assert np.all(np.linalg.eigvalsh(state) > -1e-8)
        return state

    @staticmethod
    def create_b(dist_choice, constant=True):
        if constant:
            b1 = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
            b2 = np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        else:
            rand_b = dist_choice
            b1 = np.array([0, np.sqrt(1 - (rand_b ** 2)), rand_b, 0])
            b2 = np.array([0, -1 * np.sqrt(1 - (rand_b ** 2)), rand_b, 0])
        state1 = CreateDensityMatrices.state_from_vec(b1)
        state2 = CreateDensityMatrices.state_from_vec(b2)
        assert np.allclose(state1, np.conj(state1.T))
        assert np.allclose(np.trace(state1), 1)
        assert np.allclose(state2, np.conj(state2.T))
        assert np.allclose(np.trace(state2), 1)
        assert np.all(np.linalg.eigvalsh(state1) > -1e-8)
        assert np.all(np.linalg.eigvalsh(state2) > -1e-8)
        return state1, state2

    @staticmethod
    def create_from_distribution(total_rhos: int = 1000, prop_a: float = 1/3, b_const: bool = True, 
                                 a_const: bool = False, lower: int = 0, upper: int = 1, mu_a: float = 0.5, 
                                 sigma_a: float = 0.25, mu_b: float = 0.75, sigma_b: float = 0.125):
        a_dist = truncnorm.rvs((lower-mu_a)/sigma_a, (upper-mu_a)/sigma_a, mu_a, sigma_a, size=int(total_rhos*prop_a)+2)
        b_dist = truncnorm.rvs((lower-mu_b)/sigma_b, (upper-mu_b)/sigma_b, mu_b, sigma_b, size=int(total_rhos*(1-prop_a))+2)
        a_states = []
        b_states = []

        for i in range(int(total_rhos * prop_a)):
            a_out = CreateDensityMatrices.create_a( a_dist[i], a_const)
            a_states.append(a_out)

        for i in range(int(total_rhos * (1-prop_a))):
            b1, b2 = CreateDensityMatrices.create_b(b_dist[i], b_const)
            b_states.append(b1)
            b_states.append(b2)

        return a_states, b_states


if __name__ == '__main__':
    CreateDensityMatrices.create_from_distribution(b_const=False)