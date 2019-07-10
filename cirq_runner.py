import cirq
import numpy as np
from typing import Dict, List, Union


class CirqRunner:
    """
    A class to interface the tensorflow model with Cirq.
    Takes the gate dictionaries defined in the rest of the program and returns probabilities of the results
    TODO: Add noise
    """
    def __init__(self, no_qubits: int = 4, noise_on: bool = False, noise_prob: float = 0.1, sim_repetitions: int = 100):
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
        :return: circuit, a cirq Circuit representation of that.
        """
        circuit = cirq.Circuit()
        # This is required to convert from 1 based arrays (used in gate dictionaries) to 0 based arrays
        gate_dict['qid'] = gate_dict['qid'] - 1
        gate_dict['control_qid'] = gate_dict['control_qid'] - 1
        for i in range(len(gate_dict['gate_id'])):
            circuit.append(self.read_dict(gate_dict, i), strategy=cirq.InsertStrategy.EARLIEST)
        return circuit

    def read_dict(self, gate_dict: Dict, index: int) -> cirq.Operation:
        """
        The function called for each label within the gate_dictionary. Converts that form into a cirq Gate
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
            theta_index = np.array(np.where(np.isin(gate_dict['theta_indices'], index))).reshape([])
            theta = gate_dict['theta'][theta_index]
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.XPowGate().on(target)**theta
        if label == 2:
            theta_index = np.array(np.where(np.isin(gate_dict['theta_indices'], index))).reshape([])
            theta = gate_dict['theta'][theta_index]
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.YPowGate().on(target)**theta
        if label == 3:
            theta_index = np.array(np.where(np.isin(gate_dict['theta_indices'], index))).reshape([])
            theta = gate_dict['theta'][theta_index]
            target = self.qubits[gate_dict['qid'][index]]
            yield cirq.ZPowGate().on(target)**theta
        if label == 4:
            yield cirq.IdentityGate(self.no_qubits).on(*self.qubits)
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
        if (1 in gate_dict_0['qid']) or (1 in gate_dict_1['qid']) or (1 in gate_dict_0['control_qid']) or (
                1 in gate_dict_1['control_qid']):
            raise ValueError('The controlled qubit cannot appear in the gate_dict_1 or gate_dict_0')

        circuit_0 = self.gate_dict_to_circuit(gate_dict_0)
        circuit_1 = self.gate_dict_to_circuit(gate_dict_1)
        controlled_0 = self.yield_controlled_circuit(circuit_0, self.qubits[0])
        controlled_1 = self.yield_controlled_circuit(circuit_1, self.qubits[0])

        circuit = self.gate_dict_to_circuit(gate_dict)
        circuit.append(cirq.measure(self.qubits[0], key='m0'))
        circuit.append(controlled_0)
        circuit.append(cirq.X(self.qubits[0]))
        circuit.append(controlled_1)
        circuit.append(cirq.measure(self.qubits[1], key='m1'))
        return circuit

    def calculate_probabilities(self, input_state: np.ndarray, circuit: cirq.Circuit) -> List[float]:
        """
        Simulates the given circuit and calculates the probabilities of measuring 00, 01, 10, or 11.
        :param circuit: The input circuit, should come from function above, so we have the correct measurement keys
        :param input_state: The input state to the calculation.
        :return: A list of probabilities of measuring 00, 01, 10, or 11
        """
        simulator = cirq.DensityMatrixSimulator()
        counter = np.array([0, 0, 0, 0])
        for i in range(self.reps):
            measurements = simulator.simulate(circuit, initial_state=input_state).measurements
            m0 = int(measurements['m0'])
            m1 = int(measurements['m1'])
            result = m0 << 1 | m1
            counter[result] += 1
        probs = counter / self.reps
        return probs
