import tensorflow as tf
import numpy as np
import cirq
import copy

from base_model import Model
from cirq_runner import CirqRunner
from train_model import TrainModel


class TestTraining(tf.test.TestCase):

    def test_training(self):
        trainer = TrainModel(40., 40., batch_size=2, max_epoch=5, g_epsilon=0.0001, sim_repetitions=10)
        trainer.train()