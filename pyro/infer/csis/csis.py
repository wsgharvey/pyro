from __future__ import absolute_import, division, print_function

import pyro
from pyro.infer.csis.inference import Inference
from pyro.infer.importance import Importance


class CSIS(object):
    """
    :param model: the model (callable containing Pyro primitives)

    An object for performing compiled inference: see paper
    """
    def __init__(self,
                 model,
                 optim,
                 *args,
                 **kwargs):
        self.model = model
        self.optim = optim
        self.inference = Inference(model)

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        # return self.artifact.loss(self.model, self.guide, *args, **kwargs)
        raise NotImplementedError

    def compile(self,
                num_steps,
                num_particles=8,
                *args,
                **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function
        """
        return self.inference.compile(num_steps=num_steps,
                                      optim=self.optim,
                                      num_particles=num_particles)

    def get_posterior(num_samples=None):
        """
        :num_samples: number of samples to use to approximate posterior

        returns a pyro `posterior` object which allows the creation of a `marginal` object
        """
        return Importance(model=self.model,
                          guide=self.inference.artifact,
                          num_samples=num_samples)
