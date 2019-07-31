import numpy as np

from cirq_trainer.train_model import TrainModel
from shared.create_outputs import CreateOutputs
from shared.gate_dictionaries import GateDictionaries


class TestCreateOutputs(np.testing.TestCase):

    def test_create_outputs(self):
        trainer = TrainModel(batch_size=5, max_epoch=10)
        dicts = GateDictionaries.return_new_dicts_rand_vars()
        trainer.model.set_all_dicts(*dicts)
        CreateOutputs.create_outputs(trainer.save_dir, trainer.model, trainer.test_data, trainer.runner)
