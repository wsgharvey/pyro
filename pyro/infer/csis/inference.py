from __future__ import absolute_import, division, print_function

import pyro
from pyro.infer.csis.nn import Artifact
from pyro.infer.csis.loss import Loss


class Inference(object):
    """
        object which provides functions to compile inference and draw samples
        from the artifact
    """
    def __init__(self):
        self.artifact = Artifact()
        self.loss = Loss()

    def compile(self,
                n_steps,
                optim):
        """
            trains the artifact to improve predictions
        """
        # TODO: find a way to let optim be initialised by user
        optim = optim(self.artifact.parameters(), lr=0.001)
        optim.zero_grad()

        for _ in n_steps:
            training_loss = self.loss.loss_and_grads()

            optim.step()

            optim.zero_grad()

    def posterior_samples(self):
        """
            returns weighted samples from the posterior
        """
