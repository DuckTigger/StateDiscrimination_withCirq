import copy
import datetime
import os
import sys
import time
from argparse import Namespace
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf

from cirq_backend.base_model import Model
from cirq_backend.cirq_runner import CirqRunner
from create_outputs import CreateOutputs
from shared.datasets import Datasets
from shared.gate_dictionaries import GateDictionaries
from shared.generate_data import CreateDensityMatrices


class TrainModel:

    def __init__(self, cost_error: float = 40., cost_incon: float = 10.,
                 file_loc: str = None,
                 batch_size: int = 50, max_epoch: int = 2500,
                 learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,  no_qubits: int = 4,
                 noise_on: bool = False, noise_prob: float = 0.1, sim_repetitions: int = 1000,
                 job_name: str = None, restore_loc: str = None, dicts: Tuple[Dict, Dict, Dict] = None,
                 **kwargs):

        self.dataset = Datasets(file_loc, batch_size, max_epoch)
        self.runner = CirqRunner(no_qubits, noise_on, noise_prob, sim_repetitions)
        self.model = Model(cost_error, cost_incon, self.runner)
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
            if 'zcapga1' in os.getcwd():
                self.checkpoint, self.writer = self.setup_save("/home/zcapga1/Scratch/state_discrimination/training_out")
            else:
                self.checkpoint, self.writer = self.setup_save(
                    "/home/andrew/Documents/PhD/StateDiscrimination_withCirq/training_out")
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

    def setup_save(self, save_dir: str) -> Tuple[tf.train.Checkpoint,  tf.summary.SummaryWriter]:
        """
        Sets up the tf checkpoint saver and variable writer (for tensorboard) in the directory specified.
        If directory contains a checkpoint, will restore the model from that checkpoint.
        """
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

    def save_inputs(self, namespace: Namespace):
        """
        Saves the parameters passsed there to a file, for easy restoration.
        """
        dict_copy = copy.deepcopy(self.gate_dicts)
        CreateOutputs.save_params_dicts(self.save_dir, namespace, dict_copy)

    def create_outputs(self, location: str, n_states: int = None):
        """
        When called, will take test datat and run through the circuit, creating the bar charts.
        """
        if n_states is not None:
            test_data = self.test_data.take(n_states)
        else:
            test_data = self.test_data
        CreateOutputs.create_outputs(location, self.model, test_data, self.runner)

    def train_step(self, state_batch: tf.Tensor, label_batch: tf.Tensor):
        """
        One step of training, takes a batch of states and applies gradient descent to them.
        Periodically caculates the loss, for tracking.
        """
        model = self.model
        loss = []
        for i, (state, label) in enumerate(zip(state_batch, label_batch)):
            state = state.numpy().astype(np.complex64)
            grads = model.variables_gradient_exact(state, label)
            variables = model.get_variables()
            self.optimizer.apply_gradients(zip(grads, variables))
            if i % 50:
                loss.append(model.state_to_loss(state, label))
        loss_out = tf.reduce_mean(loss)
        return loss_out

    def train(self):
        train, val, test = self.train_data, self.val_data, self.test_data
        step = 0
        with self.writer.as_default():
            for epoch in range(self.max_epoch):
                start = time.time()
                for i, batch in enumerate(train):
                    loss = self.train_step(batch[0], batch[1])
                    step += 1
                    tf.summary.scalar('Training loss', loss, step=step)
                    if i % 50 == 0:
                        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                        self.writer.flush()
                        intermediate_loc = os.path.join(self.save_dir, 'intermediate')
                        self.create_outputs(intermediate_loc, 10)
                        print('Epoch {} of {}, time for epoch is {}'.format(epoch + 1, self.max_epoch, time.time() - start))
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            outputs = os.path.join(self.save_dir, 'outputs')
            self.create_outputs(outputs)
            self.writer.flush()


if __name__ == '__main__':
    trainer = TrainModel(1., 1., batch_size=2, max_epoch=2, a_const=False, b_const=True)
    trainer.train()
