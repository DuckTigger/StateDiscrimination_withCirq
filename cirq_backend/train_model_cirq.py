import os
import time
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf

from cirq_backend.base_model import Model
from cirq_backend.cirq_runner import CirqRunner
from shared.train_model import TrainModel


class TrainModelCirq(TrainModel):

    def __init__(self, cost_error: float = 40., cost_incon: float = 10.,
                 file_loc: str = None,
                 batch_size: int = 50, max_epoch: int = 2500,
                 learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,  no_qubits: int = 4,
                 noise_on: bool = False, noise_prob: float = 0.1, sim_repetitions: int = 1000,
                 job_name: str = None, restore_loc: str = None, dicts: Tuple[Dict, Dict, Dict] = None,
                 full_dicts: bool = False, **kwargs):
        self.runner = CirqRunner(no_qubits, noise_on, noise_prob, sim_repetitions)
        self.model = Model(cost_error, cost_incon, self.runner)
        super().__init__(file_loc, batch_size, max_epoch, learning_rate, beta1, beta2, noise_prob, job_name,
                         restore_loc ,dicts, full_dicts **kwargs)

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
    trainer = TrainModelCirq(1., 1., batch_size=2, max_epoch=2, a_const=False, b_const=True)
    trainer.train()
