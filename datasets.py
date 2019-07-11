from typing import Tuple
import tensorflow as tf
import numpy as np


class Datasets:

    def __init__(self, file_loc: str = None, batch_size: int = 50, max_epoch: int = 1000):
        self.loc = file_loc
        self.batch_size = batch_size
        self.max_epoch = max_epoch

    def return_test_train_val(self) -> Tuple[tf.data.Dataset]:
        if self.loc is not None:
            return self.datasets_from_file()
        else:
            return self.generate_gdatasets()

    def datasets_from_file(self):
