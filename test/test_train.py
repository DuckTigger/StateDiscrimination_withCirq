import tensorflow as tf

from generate_data import CreateDensityMatrices
from train_model import TrainModel
from train_model_tf import TrainModelTF


class TestTraining(tf.test.TestCase):

    def test_training(self):
        trainer = TrainModel(40., 40., batch_size=20, max_epoch=2, g_epsilon=0.0001, sim_repetitions=100,
                             a_const=False, b_const=True)
        trainer.train()

    def test_correct_density_matrices(self):
        CreateDensityMatrices.create_from_distribution(10000)


class TestTrainingTF(tf.test.TestCase):

    def test_training(self):
        trainer = TrainModelTF(1., 1., batch_size=20, max_epoch=2, a_const=False, b_const=True)
        trainer.train()
