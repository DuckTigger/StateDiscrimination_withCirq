import os
import copy
from argparse import ArgumentParser
from train_model_tf import TrainModelTF
import numpy as np
import json
from typing import List


class RunAnalysisTF:

    def __init__(self, checkpoint_loc: str, no_of_states: int = None, noise_levels: List[int] = None):
        self.folder = checkpoint_loc
        if no_of_states is None:
            self.n_states = 50
        else:
            self.n_states = no_of_states
        if noise_levels is None:
            self.noise_levels = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        else:
            self.noise_levels = noise_levels

    def load_model(self) -> TrainModelTF:
        params_file = os.path.join(self.folder, 'saved_params.json')
        if not os.path.exists(os.path.join(self.folder, 'saved_params.json')):
            raise FileNotFoundError('No parameter file in this location: {}'.format(self.folder))

        with open(params_file, 'r') as f:
            params = json.load(f)
        dict_0 = copy.copy(params['gate_dict'])
        dict_1 = copy.copy(params['gate_dict_0'])
        dict_2 = copy.copy(params['gate_dict_1'])
        del params['gate_dict']
        del params['gate_dict_0']
        del params['gate_dict_1']
        dicts = (dict_0, dict_1, dict_2)
        params['dicts'] = dicts

        params['max_epoch'] = len(self.noise_levels) + 1
        params['batch_size'] = self.n_states
        params['restore_loc'] = self.folder
        params['prop_a'] = 0.5

        model = TrainModelTF(**params)
        return model

    def create_outputs(self) -> None:
        model = self.load_model()
        model.test_data = model.train_data.concatenate(model.val_data).concatenate(model.test_data)
        outputs = os.path.join(self.folder, 'outputs')
        model.create_outputs(outputs, self.n_states)

        outputs_levels = os.path.join(self.folder, 'outputs_with_noise_levels')
        for level in self.noise_levels:
            folder = os.path.join(outputs_levels, str(level).replace(r'.', r'_'))
            model.noise_prob = level
            # model.test_data = model.test_data.skip(self.n_states)
            model.create_outputs(folder, self.n_states)

    def check_probs_all_folders(self):
        output = os.path.join(self.folder, 'outputs', 'probs.npy')
        self.check_probs(output)
        output_levels = os.path.join(self.folder, 'outputs_with_noise_levels')
        noise_l = [os.path.join(output_levels, f, 'probs.npy') for f in os.listdir(output_levels) if
                   os.path.isfile(os.path.join(output_levels, f, 'probs.npy'))]
        for level in noise_l:
            self.check_probs(level)

    def check_probs(self, prob_file: str):
        output_probs = np.load(prob_file)
        check = True
        while check:
            if np.isclose(output_probs[0], 0) or np.isclose(output_probs[1], 0):
                self.create_outputs()
            else:
                check = False


def main():
    parser = ArgumentParser('Creates the outputs required. Quicker.')
    parser.add_argument('--directory', type=str, nargs=1,
                        help='The top level directory where the checkpoint folders are stored')
    parser.add_argument('--n_states', type=int, nargs='?', default=None,
                        help='The number of states in a batch for a single output.')
    parser.add_argument('--noise_levels', type=float, nargs='*', default=None,
                        help='The noise levels to validate this run with.')
    parser.add_argument('--check_for_0', action='store_true',
                        help='Set to check for 0s in the outputs and re-try (long)')
    args = parser.parse_args()

    for directory in args.directory:
        checkpoint_list = [os.path.join(directory, f) for f in os.listdir(directory) if
                           os.path.isfile(os.path.join(directory, f, 'saved_params.json'))]
        for checkpoint in checkpoint_list:
            run = RunAnalysisTF(checkpoint, no_of_states=args.n_states, noise_levels=args.noise_levels)
            run.create_outputs()
            if args.check_for_0:
                run.check_probs_all_folders()


if __name__ == '__main__':
    # run = RunAnalysisTF(r"C:\Users\Andrew Patterson\Documents\PhD\cirq_state_discrimination\checkpoints\myriad_data\tf_old_new\2019_08_19_17_52_37", 50, [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    # run.create_outputs()
    # run.check_probs_all_folders()
    main()
