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
                 guide,
                 valid_size=12,
                 *args,
                 **kwargs):
        self.model = model
        if guide is None:
            self.guide = Artifact()
        else:
            self.guide = guide
        self.args = args
        self.kwargs = kwargs
        self.valid_batch = [sample_from_prior(self.model,
                                              self.guide,
                                              *self.args,
                                              **self.kwargs)
                            for _ in range(valid_size)]
        self.iterations = 0
        self.training_losses = []
        self.valid_losses = []

    def compile(self,
                num_steps,
                optim,
                num_particles,
                valid_frequency=10):
        """
            trains the artifact to improve predictions
        """
        self.loss = Loss(num_particles=num_particles)

        # TODO: find a way to let optim be initialised by user
        # or at least let learning rate be specified outside!
        optim = optim(self.guide.parameters(), lr=1e-3)
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
            if self.iterations % valid_frequency == 0:
                valid_loss = self.loss.validation_loss(self.valid_batch,
                                                       self.guide,
                                                       *self.args,
                                                       **self.kwargs)
                self.valid_losses.append(valid_loss)
                print(" "*50, "VALIDATION LOSS IS {}".format(valid_loss))
            self.iterations += 1

        return training_loss

    def get_posterior(self, num_samples):
        """
            returns weighted samples from the posterior
        """
        # guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)
        # model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace(*args, **kwargs)
        # weight = model_trace.log_pdf()
        return pyro.infer.Importance(self.model,
                                     self.guide,
                                     num_samples)
