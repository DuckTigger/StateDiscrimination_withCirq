import cirq
import numpy as np
import copy
from typing import Dict, List, Union, Tuple

from noise_model import TwoQubitNoiseModel, two_qubit_depolarize
from generate_data import CreateDensityMatrices


class CirqRunner:
    """
    A class to interface the tensorflow model with Cirq.
    Takes the gate dictionaries defined in the rest of the program and returns probabilities of the results
    """
    def __init__(self, no_qubits: int = 4, noise_on: bool = False, noise_prob: float = 0.1, sim_repetitions: int = 10):
        self.noise_prob = noise_prob
        self.noise_on = noise_on
        self.no_qubits = no_qubits
        self.qubits = self.set_qubits(no_qubits)
        self.reps = sim_repetitions

    @property
    def noise_prob(self):
        return self.__noise_prob

    @noise_prob.setter
    def noise_prob(self, noise_prob):
        if noise_prob > 1:
            self.__noise_prob = 1
        elif noise_prob <= 0:
            self.__noise_prob = 0
        else:
            self.__noise_prob = noise_prob

    def set_qubits(self, no_qubits: int) -> List[cirq.NamedQubit]:
        """
        We initially begin the project just with named qubits, those with the fewest restrictions
        :return: qubits, a list containing qubits
        """
        qubits = [cirq.NamedQubit(str(x)) for x in range(no_qubits)]
        self.qubits = qubits
        return qubits

    def get_qubits(self):
        return self.qubits

    def gate_dict_to_circuit(self, gate_dict: Dict) -> cirq.Circuit:
        """
        Takes in the gate dictionary in the form used in teh rest of the program and returns a cirq.Circuit
        from that.
        :param gate_dict: A gate dictionary in the same form as used in the rest of the program
        :return: circuit, a cirq_trainer Circuit representation of that.
        """
        circuit = cirq.Circuit()
        for i in range(len(gate_dict['gate_id'])):
            circuit.append(self.read_dict(gate_dict, i), strategy=cirq.InsertStrategy.EARLIEST)
        return circuit

    def read_dict(self, gate_dict: Dict, index: int) -> cirq.Operation:
        """
        The function called for each label within the gate_dictionary. Converts that form into a cirq_trainer Gate
        :param gate_dict: The gate dictionary we are iterating over
        :param index: The index we are at
        :return: cirq.Operation, converted from that line in the dictionary.
        """
        label = gate_dict['gate_id'][index]

        if label == 0:
            control = self.qubits[int(gate_dict['control_qid'][np.where(np.isin(gate_dict['control_indices'], index))])]
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.CNOT(control=control, target=target)
        if label == 1:
            theta_index = np.where(np.isin(gate_dict['theta_indices'], index))[0]
            theta = gate_dict['theta'][int(theta_index)].numpy()
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.Rx(theta).on(target)
        if label == 2:
            theta_index = np.where(np.isin(gate_dict['theta_indices'], index))[0]
            theta = gate_dict['theta'][int(theta_index)].numpy()
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.Ry(theta).on(target)
        if label == 3:
            theta_index = np.where(np.isin(gate_dict['theta_indices'], index))[0]
            theta = gate_dict['theta'][int(theta_index)].numpy()
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.Rz(theta).on(target)
        if label == 4:
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.IdentityGate(1).on(target)
        if label == 5:
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.H(target)

    @staticmethod
    def yield_controlled_circuit(input_circuit: cirq.Circuit,
                                 control_qubit: Union[cirq.GridQubit, cirq.NamedQubit, cirq.LineQubit]):
        for moment in input_circuit:
            for operation in moment:
                yield operation.controlled_by(control_qubit)

    def create_full_circuit(self, gate_dict: Dict, gate_dict_0: Dict, gate_dict_1: Dict) -> cirq.Circuit:
        """
        In the state discrimination scheme, we use two different circuits based on the outcome of a
        measurement of the first qubit. Takes in the three gate dictionaries and converts them to a single circuit.
        :param gate_dict: The first gate dictionary.
        :param gate_dict_0: The gate dictionary performed if measurement of qubit 0 is 0
        :param gate_dict_1: The gate dictionary performed if measurement of qubit 0 is 1
        :return: cirq.Circuit object representing the whole circuit.
        """
        circuit_0 = self.gate_dict_to_circuit(gate_dict_0)
        circuit_1 = self.gate_dict_to_circuit(gate_dict_1)
        controlled_0 = self.yield_controlled_circuit(circuit_0, self.qubits[0])
        controlled_1 = self.yield_controlled_circuit(circuit_1, self.qubits[0])

        circuit = self.gate_dict_to_circuit(gate_dict)
        circuit.append(cirq.measure(self.qubits[0], key='m0'), strategy=cirq.InsertStrategy.NEW)
        circuit.append(controlled_1)
        circuit.append(cirq.X(self.qubits[0]))
        circuit.append(controlled_0)
        circuit.append(cirq.measure(self.qubits[1], key='m1'), strategy=cirq.InsertStrategy.NEW)
        return circuit

    def calculate_probabilities_sampling(self, input_state: np.ndarray, circuit: cirq.Circuit) -> List[float]:
        """
        Simulates the given circuit and calculates the probabilities of measuring 00, 01, 10, or 11.
        :param circuit: The input circuit, should come from function above, so we have the correct measurement keys
        :param input_state: The input state to the calculation.
        :return: A list of probabilities of measuring 00, 01, 10, or 11
        """
        simulator = cirq.DensityMatrixSimulator()
        counter = np.array([0, 0, 0, 0])
        for i in range(self.reps):
            state_in = copy.copy(input_state)
            measurements = simulator.simulate(circuit, initial_state=state_in).measurements
            m0 = int(measurements['m0'])
            m1 = int(measurements['m1'])
            result = m0 << 1 | m1
            counter[result] += 1
        probs = counter / self.reps
        return probs

    @staticmethod
    def kron_list(l):
        for i in range(len(l)):
            if i == 0:
                out = l[i]
            else:
                out = np.kron(out, l[i])
        return out

    def prob_and_set(self, state: np.ndarray, measure_qubit: int, measurement: int):
        e = np.array([[1, 0], [0, 1]]).astype(np.complex64)
        if measurement:
            o = np.array([[0, 0], [0, 1]]).astype(np.complex64)
            l = [e for _ in range(len(self.qubits))]
            l[measure_qubit] = o
            m1 = self.kron_list(l)
            rho_out = m1 @ state @ m1.T
            prob = np.trace(m1.T @ m1 @ state)
        else:
            z = np.array([[1, 0], [0, 0]]).astype(np.complex64)
            l = [e for _ in range(len(self.qubits))]
            l[measure_qubit] = z
            m0 = self.kron_list(l)
            rho_out = m0 @ state @ m0.T
            prob = np.trace(m0.T @ m0 @ state)

        rho_out = rho_out / np.trace(rho_out)
        return prob, rho_out

    def prob_2q(self, state: np.ndarray, measure_qubits: Tuple[int, ...], measurement: Tuple[int, ...]):
        e = np.array([[1, 0], [0, 1]]).astype(np.complex64)
        z = np.array([[1, 0], [0, 0]]).astype(np.complex64)
        o = np.array([[0, 0], [0, 1]]).astype(np.complex64)
        zero_one = [z, o]
        l = [e for _ in range(len(self.qubits))]
        for i, qubit in enumerate(measure_qubits):
            l[qubit] = zero_one[measurement[i]]
        m = self.kron_list(l)
        rho_out = m @ state @ m.T
        prob = np.trace(m.T @ m @ state)
        rho_out = rho_out / np.trace(rho_out)
        return prob, rho_out

    def probs_controlled_part(self, gate_dict: Dict, state_in: np.ndarray, prob: np.ndarray,
                              sim: cirq.DensityMatrixSimulator) -> Tuple[float, float]:
        if prob > 1e-8 and CreateDensityMatrices.check_state(state_in):
            circuit_0 = self.gate_dict_to_circuit(gate_dict)
            rho_0 = sim.simulate(circuit_0, initial_state=state_in).final_density_matrix
            prob_0, _ = self.prob_and_set(rho_0, 1, 0)
            prob_1, _ = self.prob_and_set(rho_0, 1, 1)
        else:
            prob_0 = 0
            prob_1 = 0
        return prob_0, prob_1

    def calculate_probabilities(self, gate_dicts: Tuple[Dict, Dict, Dict],
                                state: np.ndarray) -> List[float]:
        state_in = copy.copy(state)
        if self.noise_on:
            simulator = cirq.DensityMatrixSimulator(noise=TwoQubitNoiseModel(cirq.depolarize(4 * self.noise_prob / 5),
                                                                                     two_qubit_depolarize(self.noise_prob)))
        else:
            simulator = cirq.DensityMatrixSimulator()
        circuit_pre = self.gate_dict_to_circuit(gate_dicts[0])
        rho = simulator.simulate(circuit_pre, initial_state=state_in).final_density_matrix

        prob_0, state_0 = self.prob_and_set(rho, 0, 0)
        prob_1, state_1 = self.prob_and_set(rho, 0, 1)

        prob_00, prob_01 = self.probs_controlled_part(gate_dicts[1], state_0, prob_0, simulator)
        prob_10, prob_11 = self.probs_controlled_part(gate_dicts[2], state_1, prob_1, simulator)

        fin_00 = prob_0 * prob_00
        fin_01 = prob_0 * prob_01
        fin_10 = prob_1 * prob_10
        fin_11 = prob_1 * prob_11

        out = [fin_00, fin_01, fin_10, fin_11]
        out = [x.astype(np.float64) for x in out]
        return out

    def calculate_energy(self, u: float, v: float, gate_dict: Dict) -> float:
        if self.noise_on:
            simulator = cirq.DensityMatrixSimulator(noise=TwoQubitNoiseModel(cirq.depolarize(4 * self.noise_prob / 5),
                                                                                     two_qubit_depolarize(self.noise_prob)))
        else:
            simulator = cirq.DensityMatrixSimulator()

        circuit_pre = self.gate_dict_to_circuit(gate_dict)
        rho = simulator.simulate(circuit_pre).final_density_matrix
        measure_x0 = cirq.Circuit.from_ops([cirq.X(self.qubits[0]),
                                                    cirq.I(self.qubits[1])])
        measure_x1 = cirq.Circuit.from_ops([cirq.X(self.qubits[1]),
                                                    cirq.I(self.qubits[0])])
        measure_z0z1 = cirq.Circuit.from_ops([cirq.Z(self.qubits[0]), cirq.Z(self.qubits[1])])

        rho_x0 = simulator.simulate(measure_x0, initial_state=rho).final_density_matrix
        rho_x1 = simulator.simulate(measure_x1, initial_state=rho).final_density_matrix
        rho_z0z1 = simulator.simulate(measure_z0z1, initial_state=rho).final_density_matrix

        prob_x0, _ = self.prob_and_set(rho_x0, 0, 0)
        prob_x1, _ = self.prob_and_set(rho_x1, 1, 0)
        prob_z0, _ = self.prob_and_set(rho_z0z1, 0, 0)
        prob_z1, _ = self.prob_and_set(rho_z0z1, 1, 0)
        prob_z0z1, _ = self.prob_2q(rho_z0z1, (0, 1), (0, 0))

        energy = (u / 4)*prob_z0z1 + (v/2)*(prob_x0 + prob_x1)
        return energy

    def calculate_energy_sampling(self, u: float, v: float, gate_dict: Dict) -> float:
        if self.noise_on:
            simulator = cirq.DensityMatrixSimulator(noise=TwoQubitNoiseModel(cirq.depolarize(4 * self.noise_prob / 5),
                                                                                     two_qubit_depolarize(self.noise_prob)))
        else:
            simulator = cirq.DensityMatrixSimulator()

        circuit_pre = self.gate_dict_to_circuit(gate_dict)
        measure_x0 = cirq.Circuit.from_ops([cirq.X(self.qubits[0]),
                                                    cirq.I(self.qubits[1]), cirq.measure(self.qubits[0], key='x0')])
        measure_x1 = cirq.Circuit.from_ops([cirq.X(self.qubits[1]),
                                                    cirq.I(self.qubits[0]), cirq.measure(self.qubits[1], key='x1')])
        measure_z0z1 = cirq.Circuit.from_ops([cirq.Z(self.qubits[0]), cirq.Z(self.qubits[1]),
                                                      cirq.measure(self.qubits[0], self.qubits[1], key='z0z1')])
        circuit_x0 = circuit_pre.copy()
        circuit_x0.append(measure_x0)
        circuit_x1 = circuit_pre.copy()
        circuit_x1.append(measure_x1)
        circuit_pre.append(measure_z0z1)

        prob_x0 = simulator.run(circuit_x0, repetitions=self.reps).histogram(key='x0')[0] / self.reps
        prob_x1 = simulator.run(circuit_x1, repetitions=self.reps).histogram(key='x1')[0] / self.reps
        prob_z0z1 = simulator.run(circuit_pre, repetitions=self.reps).histogram(key='z0z1')[0] / self.reps
        energy = (u / 4)*prob_z0z1 + (v/2)*(prob_x0 + prob_x1)
        return energy
