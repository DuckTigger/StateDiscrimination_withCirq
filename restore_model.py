import os
import json
import numpy as np
from argparse import ArgumentParser
from typing import Dict
from train_model import TrainModel


class RestoreModel:

    def __init__(self):
        pass

    @staticmethod
    def load_params(model_loc: str) -> Dict:
        param_file = os.path.join(model_loc, 'saved_params.json')
        if not os.path.exists(param_file):
            raise FileNotFoundError('The parameter file is not in the model location.')
        with open(param_file) as f:
            params = json.load(f)
        return params

    @staticmethod
    def restore(model_loc: str) -> TrainModel:
        params = RestoreModel.load_params(model_loc)
        dicts = params.pop('gate_dict', None), params.pop('gate_dict_0', None), params.pop('gate_dict_1', None)
        for d in dicts:
            for key, value in d.items():
                d[key] = np.array(d[key])

        if params['job_name'] is None:
            params['job_name'] = 'restored'
        else:
            params['job_name'] = params['job_name'] + '_restored'

        trainer = TrainModel(restore_loc=model_loc, dicts=dicts, **params)
        return trainer

    @staticmethod
    def restore_and_train(model_loc: str):
        trainer = RestoreModel.restore(model_loc)
        trainer.train()

    @staticmethod
    def restore_create_outputs(model_loc: str):
        trainer = RestoreModel.restore(model_loc)
        outputs = os.path.join(model_loc, 'restored_outputs')
        trainer.create_outputs(outputs)


def main():

    parser = ArgumentParser(description='Restores the model from a given location, '
                                        'either restarts training or creates some outputs.')
    parser.add_argument('--model_loc', type=str,
                        help='The location of the tf checkpoint where the model is located. '
                             'Should also have a correctly saved parameters file.')
    parser.add_argument('--create_outputs', action='store_true',
                        help='If called, just creates outputs from the saved model, does not continue training.')
    args = parser.parse_args()

    if args.create_outputs:
        RestoreModel.restore_create_outputs(args.model_loc)
    else:
        RestoreModel.restore_and_train(args.model_loc)


if __name__ == '__main__':
    main()
