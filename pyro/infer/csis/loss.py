from __future__ import absolute_import, division, print_function

import warnings

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import torch_zeros_like
from pyro.infer.util import torch_backward
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match
from pyro.infer.csis.util import sample_from_prior

import numpy as np

"""
should provide methods to calculate loss over 1 - a number of random draws from p
                            2 - a provided batch of traces
and either calculate gradients or not bother

probably fine how it is unless it can be made a little neater - I think this functionality could be combined into one overall loss function
"""


class Loss(object):
    """
    An object to calculate an estiamte of the loss and gradients for inference
    compilation
    """
    def __init__(self,
                 args,
                 kwargs,
                 num_particles):
        """
        :num_particles: the number of particles to use for estimating the loss
        """
        # TODO: maybe put model and guide in __init__
        # put args/kwargs here too?
        super(Loss, self).__init__()
        self.num_particles = num_particles
        self.args = args
        self.kwargs = kwargs

    def _get_matched_trace(self, model_trace, guide, *args, **kwargs):
        """
            takes in a trace from the model and returns a trace of the guide, with
            observed values used as arguments and samples restricted to be the same
            as in the model trace
        """
        # set arguments to be observed values
        for name in model_trace.observation_nodes:
            kwargs[name] = model_trace.nodes[name]["value"]

        guide_trace = poutine.trace(poutine.replay(guide, model_trace)).get_trace(*args, **kwargs)

        check_model_guide_match(model_trace, guide_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        return guide_trace

    def loss(self,
             model,
             guide,
             grads=False,
             batch=None):
        """
        :returns: returns an estimate of the loss (expectation over p of -log q)
        :rtype: float

        If a batch is provided, the loss is estimated using these traces
        Otherwise, num_samples traces are generated

        If grads is True, will also calculate gradients
        """
        if batch is None:
            batch = (sample_from_prior(model, *self.args, **self.kwargs) for _ in range(self.num_particles))
            batch_size = self.num_particles
        else:
            batch_size = len(batch)

        loss = 0
        for model_trace in batch:
            guide_trace = self._get_matched_trace(model_trace, guide, *self.args, **self.kwargs)

            particle_loss = -guide_trace.log_pdf() / batch_size

            if grads:
                torch_backward(particle_loss)

            loss += particle_loss.data.numpy()[0]

        if np.isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
