import tensorflow as tf
import numpy as np
import datetime, time, sys, os

from base_model import Model
from cirq_runner import CirqRunner


class TrainModel:

    def __init__(self, cost_error: float, cost_incon: float,
                 batch_size: int = 50, max_epoch: int = 1000,
                 learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 g_epsilon: float = 1e-6, no_qubits: int = 4,
                 noise_on: bool = False, noise_prob: float = 0.1, sim_repetitions: int = 1000):
        self.runner = CirqRunner(no_qubits, noise_on, noise_prob, sim_repetitions)
        self.model = Model(cost_error, cost_incon, self.runner, g_epsilon)
        self.save_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.checkpoint_prefix = None
        self.optimizer = tf.optimizers.Adam(learning_rate, beta1, beta2)

        if sys.platform.startswith('win'):
            self.checkpoint = self.setup_save(
                "C:\\Users\\Andrew Patterson\\Documents\\PhD\\cirq_state_discrimination\\checkpoints")
        else:
            self.checkpoint = self.setup_save("/home/zcapga1/Scratch/state_discrimination/training_out")

    def setup_save(self, save_dir: str) -> tf.train.Checkpoint:
        checkpoint_dir = os.path.join(save_dir, self.save_time)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        return checkpoint

    def train_step(self, state: tf.Tensor, label: tf.Tensor):
        model = self.model
        loss = model.state_to_loss(state, label)
        grads = model.variables_gradient(loss, state, label)
        variables = model.get_variables()
        self.optimizer.apply_gradients(zip(grads, variables))

    