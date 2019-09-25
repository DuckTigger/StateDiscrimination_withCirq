import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif':['Computer Modern Roman']})
import numpy as np
import json
from typing import Dict, Tuple, Union
from argparse import Namespace

from cirq_runner import CirqRunner
from tf2_simulator_runner import TF2SimulatorRunner
from base_model import Model
from generate_data import CreateDensityMatrices


class CreateOutputs:

    def __init__(self):
        pass

    @staticmethod
    def get_final_probabilities(model: Model, test_data: tf.data.Dataset, runner: Union[CirqRunner, TF2SimulatorRunner]):
        prob_pure = []
        prob_mixed = []
        for batch in test_data:
            for state, label in zip(batch[0], batch[1]):
                state_in = state.numpy().astype(np.complex64)
                if isinstance(runner, CirqRunner):
                    check = CreateDensityMatrices.check_state(state_in)
                else:
                    check = True
                if check:
                    gate_dicts = model.return_gate_dicts()
                    measurements = runner.calculate_probabilities(gate_dicts, state_in)
                    probs = [measurements[0] + measurements[2], measurements[1], measurements[3]]
                    if label.numpy() == 0:
                        prob_pure.append(probs)
                    else:
                        prob_mixed.append(probs)
        prob_pure = np.mean(prob_pure, 0)
        prob_mixed = np.mean(prob_mixed, 0)
        return prob_pure, prob_mixed

    @staticmethod
    def create_bar_plot(save_loc: str, prob_pure: np.ndarray, prob_mixed: np.ndarray):
        x = []
        y = []
        dz = []

        classes = np.array([0, 1, 2])
        xticks = [0, 1]
        ylabs = ['$a$ States', '$b$ States', 'Inconclusive']
        xlabs = ['$a$ States', '$b$ States']
        data = np.stack([prob_pure, prob_mixed])

        for r, row in enumerate(data):
            for c, col in enumerate(row):
                x.append(xticks[r])
                y.append(classes[c])
                dz.append(col)

        z = np.zeros(len(y))
        dx = np.ones(len(y)) * 0.5
        dy = np.ones(len(y)) * 0.5

        ax3d = plt.figure().gca(projection='3d')
        colours = ['#7E57C2', '#7E57C2', '#7E57C2', '#707B7C', '#707B7C', '#707B7C']
        ax3d.bar3d(x, y, z, dx, dy, dz, color=colours, edgecolor='black')

        ax3d.axes.set_yticklabels(ylabs)
        ax3d.axes.set_yticks(classes)
        ax3d.yaxis.set_rotate_label(False)
        ax3d.yaxis.labelpad = 33
        ax3d.axes.set_ylabel(r'Output', fontsize=23, linespacing=3.5)
        ax3d.yaxis._axinfo['label']['space_factor'] = 5.0
        ax3d.set_ylim(0, 2.5)

        ax3d.axes.set_xticklabels(xlabs)
        ax3d.axes.set_xticks(xticks)
        ax3d.xaxis.set_rotate_label(False)
        ax3d.xaxis.labelpad = 17
        ax3d.axes.set_xlabel('Input states', fontsize=23, linespacing=3.5)
        ax3d.xaxis._axinfo['label']['space_factor'] = 3.0

        ax3d.set_xlim(0, 1.5)

        ax3d.axes.set_zlabel('Probability', fontsize=23)
        ax3d.zaxis.labelpad = 10

        ax3d.yaxis.set_tick_params(labelsize=19, pad=12)
        ax3d.xaxis.set_tick_params(labelsize=19)
        ax3d.zaxis.set_tick_params(labelsize=19)
        # plt.show()
        plt.savefig(os.path.join(save_loc, 'bar_graph.png'))
        np.save(os.path.join(save_loc, 'probs.npy'), data)

    @staticmethod
    def save_angles(save_loc: str, model: Model):
        variables = model.get_variables()
        variables = [v.numpy() for v in variables]
        np.save(os.path.join(save_loc, 'final_angles.npy'), variables)

    @staticmethod
    def create_outputs(save_loc: str, model: Model, test_data: tf.data.Dataset,
                       runner: Union[CirqRunner, TF2SimulatorRunner]):
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        pure, mixed = CreateOutputs.get_final_probabilities(model, test_data, runner)
        CreateOutputs.create_bar_plot(save_loc, pure, mixed)
        CreateOutputs.save_angles(save_loc, model)

    @staticmethod
    def save_params_dicts(save_loc: str, namespace: Namespace, dicts: Tuple[Dict, Dict, Dict]):
        for d in dicts:
            for key, val in d.items():
                if key == 'theta':
                    d[key] = None
                else:
                    d[key] = d[key].tolist()

        out = vars(namespace)
        out.pop('dicts', None)
        out['gate_dict'] = dicts[0]
        out['gate_dict_0'] = dicts[1]
        out['gate_dict_1'] = dicts[2]

        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        with open(os.path.join(save_loc, 'saved_params.json'), 'w') as f:
            json.dump(out, f)

if __name__ == '__main__':
    probs = np.array([[0.92672777, 0.03029269, 0.04297941], [0.26218718, 0.47460639, 0.26320657]])
    CreateOutputs.create_bar_plot(r"C:\Users\Andrew Patterson\Desktop", probs[0], probs[1])
