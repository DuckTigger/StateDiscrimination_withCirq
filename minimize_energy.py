import tensorflow as tf
import numpy as np
import sys

from cirq_runner import CirqRunner
from gate_dictionaries import GateDictionaries
from minimize_energy_model import MinimizeEnergyModel


class MinimizeEnergy():

    def __init__(self, u: float, v: float, max_epoch: int = 1000):
        self.u = u
        self.v = v
        self.runner = CirqRunner(no_qubits=2, sim_repetitions=1000)
        self.gate_dict = GateDictionaries.return_energy_min_dict()
        self.max_epoch = max_epoch
        self.model = MinimizeEnergyModel(u, v)
        self.optimizer = tf.optimizers.Adam()

    def train_step(self):
        energy = self.model.energy_fn()
        grads = self.model.gradient_fn()
        self.optimizer.apply_gradients(zip(grads, self.model.get_variables()))
        return energy

    def train(self):
        self.model.set_gate_dict(self.gate_dict)
        for i in range(self.max_epoch):
            energy = self.train_step()
            print('epoch {}, energy = {}'.format(i+1, energy))
            print('Angles: {}'.format(self.model.get_variables()))


if __name__ == '__main__':
    trainer = MinimizeEnergy(u=4, v=0.74536, max_epoch=int(sys.argv[1]))
    trainer.train()