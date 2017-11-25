from __future__ import absolute_import, division, print_function

import pyro
from pyro.infer.csis.nn import Artifact
from pyro.infer.csis.loss import Loss


class Inference(object):
    """
        object which provides functions to compile inference and draw samples
        from the artifact
    """
    def __init__(self,
                 model):
        self.model = model
        self.guide = Artifact()

    def compile(self,
                n_steps,
                optim,
                num_particles,
                *args,
                **kwargs):
        """
            trains the artifact to improve predictions
        """
        self.loss = Loss(num_particles=num_particles)

        # TODO: find a way to let optim be initialised by user
        optim = optim(self.guide.parameters())
        optim.zero_grad()

        for _ in range(n_steps):
            training_loss = self.loss.loss_and_grads(self.model,
                                                     self.guide,
                                                     *args,
                                                     **kwargs)
            print(training_loss)
            optim.step()

            optim.zero_grad()

        return training_loss

    def posterior_samples(self):
        """
            returns weighted samples from the posterior
        """
