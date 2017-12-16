from __future__ import absolute_import, division, print_function

import warnings

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import torch_zeros_like
from pyro.infer.util import torch_backward
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match
from pyro.infer.csis.prior import sample_from_prior

import numpy as np


class Loss(object):
    """
    An object to calculate an estiamte of the loss and gradients for inference
    compilation
    """
    def __init__(self,
                 num_particles):
        """
        :num_particles: the number of particles to use for estimating the loss
        """
        super(Loss, self).__init__()
        self.num_particles = num_particles

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

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the loss (expectation over p of -log q)
        :rtype: float

        Performs backward on the loss. Num_particle many samples are used to form the estimators.
        """
        loss = 0
        # for weight, guide_trace in self._get_training_traces(model, guide, *args, **kwargs):
        for _ in range(self.num_particles):
            model_trace = sample_from_prior(model, guide, *args, **kwargs)
            guide_trace = self._get_matched_trace(model_trace, guide, *args, **kwargs)

            particle_loss = -guide_trace.log_pdf() / self.num_particles

            # get gradients
            torch_backward(particle_loss)

            loss += particle_loss.data.numpy()[0]

        if np.isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss

    def validation_loss(self, validation_traces, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the loss (expectation over p of -log q)
        :rtype: float

        Estimates loss using the validation batch
        """
        loss = 0
        for model_trace in validation_traces:
            guide_trace = self._get_matched_trace(model_trace, guide, *args, **kwargs)

            particle_loss = -guide_trace.log_pdf() / len(validation_traces)

            loss += particle_loss.data.numpy()[0]

        return loss
