from __future__ import absolute_import, division, print_function

import pyro
from pyro.infer.csis.inference import Inference
from pyro.infer.importance import Importance
from pyro.infer.csis import prior

import torch


class CSIS(object):
    """
    :param model: the model (callable containing Pyro primitives)

    An object for performing compiled inference: see paper
    """
    def __init__(self,
                 model,
                 guide,
                 optim=torch.optim.Adam,
                 *args,
                 **kwargs):
        self.model = model
        self.optim = optim
        self.inference = Inference(model,
                                   guide=guide)

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

    def sample_from_prior(self, *args, **kwargs):
        return prior.sample_from_prior(self.model,
                                       *args,
                                       **kwargs)

    def get_posterior(self, num_samples):
        """
        :num_samples: number of samples to use to approximate posterior

        returns a pyro `posterior` object which allows the creation of a `marginal` object
        """
        return self.inference.get_posterior(num_samples)
