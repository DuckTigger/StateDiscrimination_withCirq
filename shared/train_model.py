import copy
import datetime
import os
import sys
import time
from argparse import Namespace
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf

from tensorflow_backend.base_model_tf import ModelTF
from tensorflow_backend.tf2_simulator_runner import TF2SimulatorRunner
from create_outputs import CreateOutputs
from shared.datasets import Datasets
from shared.gate_dictionaries import GateDictionaries


class TrainModel:

    def __init__(self, cost_error: float = 40., cost_incon: float = 10.,
                 file_loc: str = None,
                 batch_size: int = 50, max_epoch: int = 2500,
                 learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, no_qubits: int = 4,
                 noise_on: bool = False, noise_prob: float = 0.1, job_name: str = None, restore_loc: str = None,
                 dicts: Tuple[Dict, Dict, Dict] = None, **kwargs):

        self.dataset = Datasets(file_loc, batch_size, max_epoch)
        self.noise_prob = noise_prob
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.save_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.checkpoint_prefix = None
        self.save_dir = None
        self.optimizer = tf.optimizers.Adam(learning_rate, beta1, beta2)
        self.job_name = job_name
        self.restore_loc = restore_loc
        self.train_data, self.val_data, self.test_data = self.dataset.return_train_val_test(**kwargs)
        if dicts is None:
            self.gate_dicts = GateDictionaries.return_new_dicts_rand_vars()
        else:
            self.gate_dicts = GateDictionaries.fill_dicts_rand_vars(dicts)

        if sys.platform.startswith('win'):
            self.checkpoint, self.writer = self.setup_save(
                "C:\\Users\\Andrew Patterson\\Documents\\PhD\\cirq_state_discrimination\\checkpoints")
        else:
            self.checkpoint, self.writer = self.setup_save("/home/zcapga1/Scratch/state_discrimination/training_out")


