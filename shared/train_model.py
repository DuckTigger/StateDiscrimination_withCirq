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

    def __init__(self, file_loc: str = None,
                 batch_size: int = 50, max_epoch: int = 2500,
                 learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 noise_prob: float = 0.1, job_name: str = None, restore_loc: str = None,
                 dicts: Tuple[Dict, Dict, Dict] = None, full_dicts: bool = False, **kwargs):

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
            self.gate_dicts = GateDictionaries.return_new_dicts_rand_vars() if not full_dicts \
                else GateDictionaries.return_dicts_rand_vars()
        else:
            self.gate_dicts = GateDictionaries.fill_dicts_rand_vars(dicts)

        if sys.platform.startswith('win'):
            self.checkpoint, self.writer = self.setup_save(
                "C:\\Users\\Andrew Patterson\\Documents\\PhD\\cirq_state_discrimination\\checkpoints")
        else:
            if 'zcapga1' in os.getcwd():
                self.checkpoint, self.writer = self.setup_save("/home/zcapga1/Scratch/state_discrimination/training_out")
            else:
                self.checkpoint, self.writer = self.setup_save(
                    "/home/andrew/Documents/PhD/StateDiscrimination_withCirq/training_out")
    @property
    def noise_prob(self):
        return self.__noise_prob

    @noise_prob.getter
    def noise_prob(self):
        return self.__noise_prob

    @noise_prob.setter
    def noise_prob(self, noise_prob):
        if noise_prob > 1:
            self.__noise_prob = 1
            self.runner.noise_prob = 1
        elif noise_prob <= 0:
            self.__noise_prob = 0
            self.runner.noise_prob = 0
        else:
            self.__noise_prob = noise_prob
            self.runner.noise_prob = noise_prob

    @property
    def gate_dicts(self):
        return self.__gate_dicts

    @gate_dicts.getter
    def gate_dicts(self):
        return self.__gate_dicts

    @gate_dicts.setter
    def gate_dicts(self, dicts):
        self.model.set_all_dicts(*dicts)
        self.__gate_dicts = dicts

    def setup_save(self, save_dir: str) -> Tuple[tf.train.Checkpoint, tf.summary.SummaryWriter]:
        if self.job_name is None:
            checkpoint_dir = os.path.join(save_dir, self.save_time)
        else:
            checkpoint_dir = os.path.join(save_dir, self.job_name, self.save_time)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.save_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        writer = tf.summary.create_file_writer(checkpoint_dir)

        if self.restore_loc is not None:
            ckpt_path = tf.train.latest_checkpoint(self.restore_loc)
            checkpoint.restore(ckpt_path)
        return checkpoint, writer

    def reshape_vars(self):
        self.checkpoint_prefix = os.path.join(self.restore_loc, 'ckpt')
        var = self.model.get_variables()
        var = [tf.reshape(x, ()) for x in var]
        self.model.set_variables(var)
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def save_inputs(self, namespace: Namespace):
        dict_copy = copy.deepcopy(self.gate_dicts)
        CreateOutputs.save_params_dicts(self.save_dir, namespace, dict_copy)

    def create_outputs(self, location: str, n_states: int = None):
        if n_states is not None:
            test_data = self.test_data.take(n_states)
        else:
            test_data = self.test_data
        CreateOutputs.create_outputs(location, self.model, test_data, self.runner)


