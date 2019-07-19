import tensorflow as tf

from generate_data import CreateDensityMatrices
from train_model import TrainModel


class TestTraining(tf.test.TestCase):

    def test_training(self):
        trainer = TrainModel(40., 40., batch_size=20, max_epoch=2, g_epsilon=0.0001, sim_repetitions=100)
        trainer.train_data(a_const=False, b_const=True)

    def test_correct_density_matrices(self):
        CreateDensityMatrices.create_from_distribution(10000)

    def test_new_training(self):
        trainer = TrainModel(40., 40., batch_size=1, max_epoch=2, g_epsilon=0.0001, sim_repetitions=100)
        trainer.train_data(a_const=False, b_const=True)
