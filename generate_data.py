import numpy as np
from scipy.stats import truncnorm
import os
import json


class CreateDensityMatrices:

    def __init__(self):
        pass

    @staticmethod
    def state_from_vec(vec: np.ndarray):
        zero = np.array([[1, 0], [0, 0]])
        zero_zero = np.kron(zero, zero)
        rho = np.einsum('i,j->ij', vec, np.conj(vec))
        rho = rho / np.trace(rho)
        state = np.kron(zero_zero, rho)
        state = state / np.trace(state)
        return state

    @staticmethod
    def create_a(out, dist_choice, a_const=False):
        if a_const:
            psi_a = np.array([np.sqrt(1 - (0.25 ** 2)), 0, 0.25, 0])
        else:
            rand_a = dist_choice
            psi_a = np.array([np.sqrt(1 - (rand_a ** 2)), 0, rand_a, 0])
        state = CreateDensityMatrices.state_from_vec(psi_a)
        out.append(state)
        assert np.allclose(state, np.conj(state.T))
        assert np.allclose(np.trace(state), 1)
        return out

    @staticmethod
    def create_b(out, dist_choice, constant=True):
        if constant:
            b1 = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
            b2 = np.array([0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0])
        else:
            rand_b = dist_choice
            b1 = np.array([0, np.sqrt(1 - (rand_b ** 2)), rand_b, 0])
            b2 = np.array([0, -1 * np.sqrt(1 - (rand_b ** 2)), rand_b, 0])
        state1 = CreateDensityMatrices.state_from_vec(b1)
        state2 = CreateDensityMatrices.state_from_vec(b2)
        out.append(state1)
        out.append(state2)
        assert np.allclose(state1, np.conj(state1.T))
        assert np.allclose(np.trace(state1), 1)
        assert np.allclose(state2, np.conj(state2.T))
        assert np.allclose(np.trace(state2), 1)
        return out

    @staticmethod
    def create_from_distribution(total_rhos: int = 1000, prop_a: float = 1/3, b_const: bool = True,
                                 a_const: bool = False, lower: int = 0, upper: int = 1, mu: float = 0.5,
                                 sigma: float = 0.25):
        a_dist = truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, mu, sigma, size=int(total_rhos*prop_a)+2)
        b_dist = truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, 0.75, 0.125, size=int(total_rhos*(1-prop_a))+2)
        a_states = []
        b_states = []

        for i in range(int(total_rhos * prop_a)):
            a_states = CreateDensityMatrices.create_a(a_states, a_dist[i], a_const)

        for i in range(int(total_rhos * (1-prop_a))):
            b_states = CreateDensityMatrices.create_b(b_states, b_dist[i], b_const)

        return a_states, b_states
