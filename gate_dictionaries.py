import numpy as np
from typing import List, Dict, Tuple
import tensorflow as tf


class GateDictionaries:

    def __init__(self):
        pass

    @staticmethod
    def return_empty_dicts():
        """
        The standard form of gate dictionaries used in the old scheme - modified slightly.
        :return: The three gate dictionaries hard-coded here.
        """
        gate_dict = {
            'gate_id': np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0]),
            'theta': None,
            'theta_indices': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            'control_qid': np.array([0, 1, 2, 3]),
            'control_indices': np.array([12, 13, 14, 15]),
            'qid': np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0])
        }

        gate_dict_0 = {
            'gate_id': np.array([4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0]),
            'theta': None,
            'theta_indices': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'control_qid': np.array([1, 2, 3]),
            'control_indices': np.array([10, 11, 12]),
            'qid': np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1])
        }

        gate_dict_1 = {
            'gate_id': np.array([4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0]),
            'theta': None,
            'theta_indices': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'control_qid': np.array([1, 2, 3]),
            'control_indices': np.array([10, 11, 12]),
            'qid': np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3, 1])
        }
        return gate_dict, gate_dict_0, gate_dict_1

    @staticmethod
    def build_new_dicts():

        gate_dict = GateDictionaries.build_dict(gate_id=np.array([0, 0, 0, 0, 1, 1, 3, 3, 1, 1]),
                                                control=np.array([3, 3, 2, 2]),
                                                qid=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
        gate_dict_0 = GateDictionaries.build_dict(gate_id=np.array([0, 0, 1, 3, 1, 4]),
                                                  control=np.array([3, 2]),
                                                  qid=np.array([1, 1, 1, 1, 1, 0]))

        gate_dict_1 = GateDictionaries.build_dict(gate_id=np.array([0, 0, 1, 3, 1, 4]),
                                                  control=np.array([3, 2]),
                                                  qid=np.array([1, 1, 1, 1, 1, 0]))
        return gate_dict, gate_dict_0, gate_dict_1

    @staticmethod
    def build_dict(gate_id: np.ndarray, control: np.ndarray, qid: np.ndarray, theta: np.ndarray = None):
        """
        Creates a dictionary from the parameters given easily creates the theta indices and control indices
        :param gate_id:
        :param control:
        :param qid:
        :param theta:
        :return:
        """
        gate_dict = {
            'gate_id': gate_id,
            'theta': theta,
            'theta_indices': np.where((gate_id != 0) & (gate_id != 4) & (gate_id != 5))[0],
            'control_qid': control,
            'control_indices': np.where(gate_id == 0)[0],
            'qid': qid
        }
        return gate_dict

    @staticmethod
    def build_three_dicts(gate_id: np.ndarray, qid: np.ndarray, control_qid: np.ndarray, theta: List = None):
        """
        Builds three nearly identical gate_dicts based oin the structure for the largest passed to gate_id,
        i.e. the dicts will be the same for all three dicts on qubits 2, 3, 4, but will onlly contain operations on
        qubit 1 in the first dict.
        :param gate_id: The gate list to be perfomed CNOT:0, Rx:1, Ry:2, Rz:3, I:4, H:5
        :param control_qid: The control qubit id
        :param qid: The target qubit id for controllled and single qubit gates
        :param theta: If given, the variables to use, if none, allows theta to be filled later
        :return: gate_dict, gate_dict_0, gate_dict_1
        """

        control_indices = np.where(gate_id == 0)[0]
        rm_control = control_indices[np.where(control_qid == 0)]
        qid_rm_ctrl = np.delete(qid, rm_control)
        gate_id_rm_ctrl = np.delete(gate_id, rm_control)
        # Hack to return the removed CNOT
        qid_post = np.append(qid_rm_ctrl[np.where(qid_rm_ctrl != 0)], 3)
        qid_post = np.append(qid_post, 0)
        gate_id_post = np.append(gate_id_rm_ctrl[np.where(qid_rm_ctrl != 0)], 0)
        gate_id_post = np.append(gate_id_post, 4)
        control_qid_post = control_qid[np.where(control_qid != 0)]

        if theta is None:
            theta0 = None
            theta1 = None
            theta2 = None
        else:
            rot_pre = len(np.where(gate_id != 0)[0])
            rot_post = len(np.where(gate_id_post != 0)[0])
            theta0 = theta[:rot_pre]
            theta1 = theta[rot_pre:rot_pre + rot_post]
            theta2 = theta[rot_pre + rot_post:rot_pre + 2*rot_post]

        gate_dict = GateDictionaries.build_dict(gate_id, control_qid, qid, theta0)
        gate_dict_0 = GateDictionaries.build_dict(gate_id_post, control_qid_post, qid_post, theta1)
        gate_dict_1 = GateDictionaries.build_dict(gate_id_post, control_qid_post, qid_post, theta2)

        return gate_dict, gate_dict_0, gate_dict_1

    @staticmethod
    def return_standard_dicts(theta: List = None):
        """
        Uses the dictionary creator to build the dictionaries hard coded above
        :param theta:
        :return:
        """
        gate_id = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0])
        qid = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0])
        control_qid = np.array([0, 1, 2, 3])
        gate_dict, gate_dict_0, gate_dict_1 = GateDictionaries.build_three_dicts(gate_id, qid, control_qid, theta)
        return gate_dict, gate_dict_0, gate_dict_1

    @staticmethod
    def return_short_dicts(theta: List = None):
        gate_id = np.array([1, 1, 1, 0])
        qid = np.array([0, 1, 2, 3])
        control_qid = np.array([0, 1])
        gate_dict, gate_dict_0, gate_dict_1 = GateDictionaries.build_three_dicts(gate_id, qid, control_qid, theta)
        return gate_dict, gate_dict_0, gate_dict_1

    @staticmethod
    def fill_dicts_rand_vars(dicts: Tuple[Dict, Dict, Dict]) -> Tuple[Dict, Dict, Dict]:
        for d in dicts:
            for key, val in d.items():
                if type(val) is list:
                    d[key] = np.array(d[key])

        th0 = len(np.where(dicts[0]['gate_id'] != 0)[0])
        th1 = len(np.where(dicts[1]['gate_id'] != 0)[0])
        th2 = len(np.where(dicts[2]['gate_id'] != 0)[0])
        rand_th = [np.random.rand() * 4 * np.pi for _ in range(th0 + th1 + th2)]
        variables = [tf.Variable(x, dtype=tf.float32, name='theta_{}'.format(i)) for i, x in enumerate(rand_th)]

        dicts[0]['theta'] = [x for x in variables[:th0]]
        dicts[1]['theta'] = [x for x in variables[th0:th0 + th1]]
        dicts[2]['theta'] = [x for x in variables[th0 + th1:th0 + th1 + th2]]
        return dicts

    @staticmethod
    def return_short_dicts_ran_vars():
        gate_dict, gate_dict_0, gate_dict_1 = GateDictionaries.return_short_dicts()
        return GateDictionaries.fill_dicts_rand_vars((gate_dict, gate_dict_0, gate_dict_1))

    @staticmethod
    def return_new_dicts_rand_vars():
        gate_dict, gate_dict_0, gate_dict_1 = GateDictionaries.build_new_dicts()
        return GateDictionaries.fill_dicts_rand_vars((gate_dict, gate_dict_0, gate_dict_1))

    @staticmethod
    def return_dicts_rand_vars():
        """
        Uses the above function to fill three dictionaries with random thetas.
        :return:
        """
        gate_dict, gate_dict_0, gate_dict_1 = GateDictionaries.return_standard_dicts()
        return GateDictionaries.fill_dicts_rand_vars((gate_dict, gate_dict_0, gate_dict_1))


def test():
    gatedicts = GateDictionaries()
    dicts_st = gatedicts.return_empty_dicts()
    dicts_gen = gatedicts.return_standard_dicts()

    for dict_st, dict_gen in zip(dicts_st, dicts_gen):
        for key, value in dict_st.items():
            np.testing.assert_array_equal(value, dict_gen[key])


if __name__ == '__main__':
    test()
