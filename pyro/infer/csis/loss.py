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


class Loss(object):
    """
    An object to calculate an estimate of the loss (and gradients if desired)
    for inference compilation
    """
    def __init__(self,
                 model,
                 guide
                 args,
                 kwargs,
                 num_particles,
                 cuda):
        """
        :num_particles: the number of particles to use for estimating the loss
        """
        super(Loss, self).__init__()
        self.model = model
        self.guide = guide
        self.args = args
        self.kwargs = kwargs
        self.num_particles = num_particles
        self.cuda = cuda

    def _get_matched_trace(self, model_trace, guide):
        """
            takes in a trace from the model and returns a trace of the guide, with
            observed values used as arguments and samples restricted to be the same
            as in the model trace
        """
        updated_kwargs = self.kwargs
        for name in model_trace.observation_nodes:
            if not self.cuda:
                updated_kwargs[name] = model_trace.nodes[name]["value"]
            else:
                updated_kwargs[name] = model_trace.nodes[name]["value"].cuda()

        guide_trace = poutine.trace(poutine.replay(guide, model_trace)).get_trace(*self.args, **updated_kwargs)

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

        If grads is True, will also call `torch_backward` on loss
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
