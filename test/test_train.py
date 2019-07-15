import tensorflow as tf
import numpy as np
import cirq
import copy

from base_model import Model
from cirq_runner import CirqRunner
from train_model import TrainModel
from generate_data import CreateDensityMatrices


class TestTraining(tf.test.TestCase):

    def test_training(self):
        trainer = TrainModel(40., 40., batch_size=20, max_epoch=2, g_epsilon=0.0001, sim_repetitions=100)
        trainer.train(a_const=False, b_const=True)

    def test_correct_density_matrices(self):
        CreateDensityMatrices.create_from_distribution(10000)

    def test_new_training(self):
        trainer = TrainModel(40., 40., batch_size=1, max_epoch=2, g_epsilon=0.0001, sim_repetitions=100)
        trainer.train_new(a_const=False, b_const=True)