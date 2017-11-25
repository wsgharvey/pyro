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
        self.validation_set = None
        self.iterations = 0
        self.training_losses = []
        self.validation_losses = []

    def compile(self,
                num_steps,
                optim,
                num_particles,
                validation_frequency=10,
                *args,
                **kwargs):
        """
            trains the artifact to improve predictions
        """
        self.loss = Loss(num_particles=num_particles)

        # TODO: find a way to let optim be initialised by user
        optim = optim(self.guide.parameters(), lr=0.1)
        optim.zero_grad()

        for _ in range(num_steps):
            training_loss = self.loss.loss_and_grads(self.model,
                                                     self.guide,
                                                     *args,
                                                     **kwargs)

            optim.step()

            optim.zero_grad()

            print("LOSS: {}".format(training_loss))
            self.iterations += 1
            self.training_losses.append(training_loss)
            if self.iterations % validation_frequency == 0:
                pass

        return training_loss

    def sample_from_prior(self):
        """
            returns samples from the unconditioned model
        """

    def posterior_samples(self):
        """
            returns weighted samples from the posterior
        """
