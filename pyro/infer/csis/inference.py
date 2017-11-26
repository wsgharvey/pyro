from __future__ import absolute_import, division, print_function

import pyro
from pyro.infer.csis.nn import Artifact
from pyro.infer.csis.loss import Loss
from pyro.infer.csis.prior import sample_from_prior


class Inference(object):
    """
        object which provides functions to compile inference and draw samples
        from the artifact
    """
    def __init__(self,
                 model,
                 validation_size=12,
                 *args,
                 **kwargs):
        self.model = model
        self.guide = Artifact()
        self.args = args
        self.kwargs = kwargs
        self.validation_batch = [sample_from_prior(self.model,
                                                   self.guide,
                                                   *self.args,
                                                   **self.kwargs)
                                 for _ in range(validation_size)]
        self.iterations = 0
        self.training_losses = []
        self.validation_losses = []

    def compile(self,
                num_steps,
                optim,
                num_particles,
                validation_frequency=10):
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
                                                     *self.args,
                                                     **self.kwargs)

            optim.step()

            optim.zero_grad()

            print("LOSS: {}".format(training_loss))
            self.training_losses.append(training_loss)
            if self.iterations % validation_frequency == 0:
                valid_loss = self.loss.validation_loss(self.validation_batch,
                                                       self.guide,
                                                       *self.args,
                                                       **self.kwargs)
                print("VALIDATION LOSS IS {}".format(valid_loss))

            self.iterations += 1

        return training_loss

    def posterior_samples(self):
        """
            returns weighted samples from the posterior
        """
