import numpy as np
from scipy.stats import truncnorm, uniform
import os
import json
from typing import Tuple, List


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
        zero = np.array([[1, 0], [0, 0]]).astype(np.complex64)
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
        :param dist_choice: the random number from the distribution
        :param a_const:
        :return:
        """
        if a_const:
            psi_a = np.array([np.sqrt(1 - (0.25 ** 2)), 0, 0.25, 0]).astype(np.complex64)
        else:
            rand_a = dist_choice
            psi_a = np.array([np.sqrt(1 - (rand_a ** 2)), 0, rand_a, 0]).astype(np.complex64)
        state = CreateDensityMatrices.state_from_vec(psi_a)
        return state

    @staticmethod
    def create_b(dist_choice, constant=True):
        if constant:
            b1 = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]).astype(np.complex64)
            b2 = np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0]).astype(np.complex64)
        else:
            rand_b = dist_choice
            b1 = np.array([0, np.sqrt(1 - (rand_b ** 2)), rand_b, 0]).astype(np.complex64)
            b2 = np.array([0, -1 * np.sqrt(1 - (rand_b ** 2)), rand_b, 0]).astype(np.complex64)
        state1 = CreateDensityMatrices.state_from_vec(b1)
        state2 = CreateDensityMatrices.state_from_vec(b2)
        return state1, state2

    @staticmethod
    def check_state(state):
        if not np.allclose(state, np.conj(state.T)):
            return False
        if not np.allclose(np.trace(state), 1):
            return False
        if not np.all(np.linalg.eigvalsh(state) > -1e-7):
            return False
        return True

    @staticmethod
    def create_random_states(total_states: int = 1000, lower: int = 0, upper: int = 1,
                             mu_a: float = 0.5, sigma_a: float = 0.15):
        if total_states == 2:
            dist = uniform.rvs(size=8)
            total_states = 8
        else:
            dist = truncnorm.rvs((lower - mu_a) / sigma_a,  (upper - mu_a) / sigma_a, mu_a, sigma_a,
                                  size=4 * total_states + 2)
        a_states = []
        b_states = []

        for i in range(0, total_states, 8):
            a_out = CreateDensityMatrices.create_random(dist[i:i+4], b=False)
            b_out = CreateDensityMatrices.create_random(dist[i+4:i+8], b=True)
            if CreateDensityMatrices.check_state(a_out):
                a_states.append(a_out)
            if CreateDensityMatrices.check_state(b_out):
                b_states.append(b_out)
        return a_states, b_states

    @staticmethod
    def create_random(dist, b: bool = False):
        phi = np.array(dist).astype(np.complex64)
        if b:
            phi[1] *= 1j
            phi[2] *= 1j
        state = CreateDensityMatrices.state_from_vec(phi)
        return state

    @staticmethod
    def create_from_distribution(total_states: int = 1000, prop_a: float = 1 / 2, b_const: bool = True,
                                 a_const: bool = False, lower: int = 0, upper: int = 1,
                                 mu_a: float = 0.5, sigma_a: float = 0.25,
                                 mu_b: float = 0.75, sigma_b: float = 0.125) -> Tuple[List, List]:
        a_dist = truncnorm.rvs((lower-mu_a) / sigma_a, (upper-mu_a) / sigma_a, mu_a, sigma_a,
                               size=int(total_states * prop_a * 2) + 2)
        b_dist = truncnorm.rvs((lower-mu_b) / sigma_b, (upper-mu_b) / sigma_b, mu_b, sigma_b,
                               size=int(total_states * (1 - prop_a)) + 2)
        a_states = []
        b_states = []

        for i in range(int(total_states * prop_a * 2)):
            a_out = CreateDensityMatrices.create_a(a_dist[i], a_const)
            if CreateDensityMatrices.check_state(a_out):
                a_states.append(a_out)

        for i in range(int(total_states * (1 - prop_a))):
            b1, b2 = CreateDensityMatrices.create_b(b_dist[i], b_const)
            if (CreateDensityMatrices.check_state(b1) and
                    CreateDensityMatrices.check_state(b2)):
                b_states.append(b1)
                b_states.append(b2)

        return a_states, b_states

    @staticmethod
    def save_to_file(file_name: str, total_states: int = 50000, prop_a: float = 1 / 3, b_const: bool = True,
                     a_const: bool = False, lower: int = 0, upper: int = 1,
                     mu_a: float = 0.5, sigma_a: float = 0.25, mu_b: float = 0.75,
                     sigma_b: float = 0.125) -> None:
        a_states, b_states = CreateDensityMatrices.create_from_distribution(total_states, prop_a, b_const,
                                                                            a_const, lower, upper, mu_a,
                                                                            sigma_a, mu_b, sigma_b)
        file_name = 'new_' + file_name
        out = dict()
        # Cast to real as atm we don't have complex values in incoming states
        # TODO:fix for future, add real and imag parts as separate lists
        a_states = [np.real(x).tolist() for x in a_states]
        b_states = [np.real(x).tolist() for x in b_states]
        out['0'] = a_states
        out['1'] = b_states
        with open(os.path.join(os.path.dirname(__file__), 'data', file_name), 'w') as f:
            json.dump(out, f)


if __name__ == '__main__':
    CreateDensityMatrices.save_to_file('mua05_sigmaa025_constb.json')
