from typing import Tuple, List
import tensorflow as tf
import os
import json
import numpy as np

from generate_data import CreateDensityMatrices


class Datasets:
    """
    A class for interacting with the saved datasets from the old scheme, and for generating new ones on-the-fly.
    """
    def __init__(self, file_name: str = None, batch_size: int = 50, max_epoch: int = 1000):
        self.f_name = file_name
        self.batch_size = batch_size
        self.max_epoch = max_epoch

    def return_train_val_test(self, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        If the file_name is given in the class initializer we work with a saved dataset, oitherwise we generate
        a new one according to the kwargs
        :param kwargs: See those in CreateDensityMatrices.create_from_distributioun()
        :return: Three tf.Datasets, train, val, test
        """
        if self.f_name is not None:
            dataset, length = self.read_from_file()
        else:
            dataset, length = self.generate_datasets(**kwargs)
        return self.split_and_batch_dataset(dataset, length)

    def generate_datasets(self, prop_a: float = 1/3, b_const: bool = True,
                          a_const: bool = False, lower: int = 0, upper: int = 1,
                          mu_a: float = 0.5, sigma_a: float = 0.25, mu_b: float = 0.75,
                          sigma_b: float = 0.125) -> Tuple[tf.data.Dataset, int]:
        """
        Generates datasets, uses the same form as the dataset from file loader to keep things consistent.
        For params see CreateDensityMatrices.create_from_distribution
        :return: dataset: the full tf.Dataset, and its length.
        """
        states = CreateDensityMatrices.create_from_distribution(self.batch_size * self.max_epoch * 100, prop_a, b_const,
                                                                a_const, lower, upper, mu_a, sigma_a, mu_b, sigma_b)
        states_set = []
        labels_set = []
        for i, st in enumerate(states):
            label_vec = np.full(fill_value=i, shape=np.shape(st)[0])
            states_set.extend(st)
            labels_set.extend(label_vec)
        states_set = tf.constant(states_set, dtype=tf.complex64)
        dataset = tf.data.Dataset.from_tensor_slices((states_set, labels_set))
        return dataset, len(labels_set)

    def read_from_file(self) -> Tuple[tf.data.Dataset, int]:
        if 'new_' in self.f_name:
            return self.read_from_file_new()
        else:
            return self.read_from_file_old()

    def read_from_file_new(self) -> Tuple[tf.data.Dataset, int]:
        with open(os.path.join(os.path.dirname(__file__), "data", self.f_name)) as f:
            data = json.load(f)

        states_set = []
        labels_set = []
        for key, val in data.items():
            states = [np.array(v).astype(np.complex64) for v in val]
            states_set.extend(states)
            label = int(key)
            label_vec = np.full(fill_value=label, shape=np.shape(states)[0])
            labels_set.extend(label_vec)
        dataset = tf.data.Dataset.from_tensor_slices((states_set, labels_set))
        return dataset, len(labels_set)

    def read_from_file_old(self) -> Tuple[tf.data.Dataset, int]:
        """
        Reads a data file from this package, created by the old scheme.
        :return: datset: the dataset in the file, and its length
        """
        with open(os.path.join(os.path.dirname(__file__), "data", self.f_name)) as f:
            data = json.load(f)

        states_set = []
        labels_set = []
        for data_class in data['QClasses']:
            keys = ['test_states', 'train_states', 'validation_states']
            states = [np.array(data_class[key]).astype(np.complex64) for key in keys]
            states = [item for sublist in states for item in sublist]
            label = data_class['label']
            label_vec = np.full(fill_value=label, shape=np.shape(states)[0])
            states_set.extend(states)
            labels_set.extend(label_vec)
        dataset = tf.data.Dataset.from_tensor_slices((states_set, labels_set))
        return dataset, len(labels_set)

    def split_and_batch_dataset(self, dataset: tf.data.Dataset, length: int) -> Tuple[
                                tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Takes a full dataset, shuffles it and splits it into test, val and train sets.
        If the dataset is not long enough for trining it is repeated and re-shuffled.
        :param dataset: The tf.Dataset to manipulate
        :param length: its length
        :return: train, val, test: Training, Validation and Test datasets
        """
        prop_train = int(0.7 * length)
        shortfall = int(prop_train / self.max_epoch * self.batch_size)
        prop_val = int(0.2 * length)
        prop_test = int(0.1 * length)
        dataset = dataset.shuffle(1000)

        if shortfall < 1:
            dataset = dataset.repeat(int(1/shortfall))
            dataset = dataset.shuffle(1000)

        train = dataset.take(prop_train)
        test = dataset.skip(prop_train)
        val = test.skip(prop_val)
        test = test.take(prop_test)

        [train, val, test] = [x.batch(self.batch_size, drop_remainder=True) for x in [train, val, test]]
        return train, val, test


if __name__ == '__main__':
    d = Datasets('sigma005_mu05_bconst.json')
    d.read_from_file()