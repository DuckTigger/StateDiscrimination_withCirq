import os
import time
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf

from tensorflow_backend.base_model_tf import ModelTF
from tensorflow_backend.tf2_simulator_runner import TF2SimulatorRunner
from shared.train_model import TrainModel


class TrainModelTF(TrainModel):

    def __init__(self, cost_error: float = 40., cost_incon: float = 10.,
                 file_loc: str = None,
                 batch_size: int = 50, max_epoch: int = 2500,
                 learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, no_qubits: int = 4,
                 noise_on: bool = False, noise_prob: float = 0.1, job_name: str = None, restore_loc: str = None,
                 dicts: Tuple[Dict, Dict, Dict] = None, full_dicts: bool = False, **kwargs):

        self.runner = TF2SimulatorRunner(no_qubits, noise_on, noise_prob)
        self.model = ModelTF(cost_error, cost_incon, self.runner)
        super().__init__(file_loc, batch_size, max_epoch, learning_rate, beta1, beta2, noise_prob, job_name,
                         restore_loc, dicts, full_dicts, **kwargs)

    @tf.function
    def caclulate_loss(self, batch: tf.data.Dataset):
        model = self.model
        loss_in = tf.fill((self.batch_size,), tf.constant(0.))
        batch_out, loss = tf.map_fn(lambda x: model.loss_fn(x[0], x[1]), (batch, loss_in))
        loss_out = tf.reduce_mean(loss)
        return loss_out

    @tf.function
    def train_step(self, batch: tf.data.Dataset):
        model = self.model
        loss_in = tf.fill((self.batch_size,), tf.constant(0.))
        grads_in = tf.stack(np.full((self.batch_size, len(self.model.get_variables())), 0.).astype(np.float32))
        grads, batch_out, _ = tf.map_fn(lambda x: model.variables_gradient_exact(x[0], x[1], x[2]),
                                        (grads_in, batch, loss_in))
        return grads

    def train(self):
        train, val, test = self.train_data, self.val_data, self.test_data
        step = 0
        with self.writer.as_default():
            for epoch in range(self.max_epoch):
                start = time.time()
                for i, batch in enumerate(train):
                    grads_out = self.train_step(batch)
                    grads_out = tf.reduce_sum(grads_out, 0)
                    self.optimizer.apply_gradients(zip(grads_out, self.model.get_variables()))
                    step += 1
                    if i % 50 ==0:
                        loss_out = self.caclulate_loss(batch)
                        tf.summary.scalar('Training loss', loss_out, step)
                    if i % 100 == 0:
                        intermediate_loc = os.path.join(self.save_dir, 'intermediate')
                        self.create_outputs(intermediate_loc, 100)
                        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                        self.writer.flush()
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            outputs = os.path.join(self.save_dir, 'outputs')
            self.create_outputs(outputs, 250)
            self.writer.flush()


if __name__ == '__main__':
    trainer = TrainModelTF(40., 40., batch_size=2, max_epoch=2, a_const=False, b_const=True)
    trainer.train()
