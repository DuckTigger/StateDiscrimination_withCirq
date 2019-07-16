import tensorflow as tf
import numpy as np
import datetime, time, sys, os
from typing import Tuple

from base_model import Model
from cirq_runner import CirqRunner
from datasets import Datasets
from gate_dictionaries import GateDictionaries
from generate_data import CreateDensityMatrices


class TrainModel:

    def __init__(self, cost_error: float, cost_incon: float,
                 file_loc: str = None,
                 batch_size: int = 50, max_epoch: int = 1000,
                 learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 g_epsilon: float = 1e-6, no_qubits: int = 4,
                 noise_on: bool = False, noise_prob: float = 0.1, sim_repetitions: int = 1000):

        self.dataset = Datasets(file_loc, batch_size, max_epoch)
        self.runner = CirqRunner(no_qubits, noise_on, noise_prob, sim_repetitions)
        self.model = Model(cost_error, cost_incon, self.runner, g_epsilon)
        self.max_epoch = max_epoch
        self.save_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.checkpoint_prefix = None
        self.optimizer = tf.optimizers.Adam(learning_rate, beta1, beta2)

        if sys.platform.startswith('win'):
            self.checkpoint, self.writer = self.setup_save(
                "C:\\Users\\Andrew Patterson\\Documents\\PhD\\cirq_state_discrimination\\checkpoints")
        else:
            self.checkpoint, self.writer = self.setup_save("/home/zcapga1/Scratch/state_discrimination/training_out")

    def setup_save(self, save_dir: str) -> Tuple[tf.train.Checkpoint,  tf.summary.SummaryWriter]:
        checkpoint_dir = os.path.join(save_dir, self.save_time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        writer = tf.summary.create_file_writer(checkpoint_dir)
        return checkpoint, writer

    # @tf.function
    def train_step(self, state_in: tf.Tensor, label_in: tf.Tensor):
        model = self.model
        for state, label in zip(state_in, label_in):
            loss = model.state_to_loss(state, label)
            grads = model.variables_gradient(loss, state, label)
            variables = model.get_variables()
            self.optimizer.apply_gradients(zip(grads, variables))
        return loss

    def train(self, **kwargs):
        # gate_dicts = GateDictionaries().return_dicts_rand_vars()
        gate_dicts = GateDictionaries().return_short_dicts_ran_vars()
        self.model.set_all_dicts(gate_dicts[0], gate_dicts[1], gate_dicts[2])
        train, val, test = self.dataset.return_train_val_test(**kwargs)

        for epoch in range(self.max_epoch):
            start = time.time()
            for batch in train:
                loss = self.train_step(batch[0], batch[1])
                tf.summary.scalar('Training loss', loss, epoch)
                tf.summary.write('loss{}'.format(epoch), loss)

            if epoch % 10 == 0:
                tf.summary.scalar('Training loss', loss, epoch)
                tf.summary.write('loss{}'.format(epoch), loss)
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print('Epoch {} of {}, time for epoch is {}'.format(epoch + 1, self.max_epoch, time.time() - start))

        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def train_step_new(self, state_batch: tf.Tensor, label_batch: tf.Tensor):
        model = self.model
        loss = []
        for state, label in zip(state_batch, label_batch):
            state = state.numpy()
            if CreateDensityMatrices.check_state(state):
                loss.append(model.state_to_loss(state, label))
                grads = model.varibles_gradient_exact(state, label)
                variables = model.get_variables()
                self.optimizer.apply_gradients(zip(grads, variables))
        loss_out = tf.reduce_mean(loss)
        return loss_out

    def train_new(self, **kwargs):
        gate_dicts = GateDictionaries.return_new_dicts_rand_vars()
        self.model.set_all_dicts(gate_dicts[0], gate_dicts[1], gate_dicts[2])
        train, val, test = self.dataset.return_train_val_test(**kwargs)

        with self.writer.as_default():
            for epoch in range(self.max_epoch):
                start = time.time()
                for batch in train:
                    loss = self.train_step_new(batch[0], batch[1])
                    tf.summary.scalar('Training loss', loss, epoch)
                    self.writer.flush()

                if epoch % 10 == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                    print('Epoch {} of {}, time for epoch is {}'.format(epoch + 1, self.max_epoch, time.time() - start))
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)


if __name__ == '__main__':
    trainer = TrainModel(40., 40., batch_size=20, max_epoch=2500)
    trainer.train_new(a_const=False, b_const=True)
