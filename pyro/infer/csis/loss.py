from __future__ import absolute_import, division, print_function

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import torch_zeros_like
from pyro.infer.util import torch_backward
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match
from pyro.infer.csis.model_traces import get_trace_from_prior

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

    def _get_traces(self, model, guide, *args, **kwargs):
        """ runs the model to generate traces
            then runs the guide against each trace and computes the loss
        """
        print(get_trace_from_prior(model, guide, *args, **kwargs))

        for _ in range(self.num_particles):

            # take a trace from the model and then take a trace from the guide with the samples constrained to be the same
            model_trace = poutine.trace(model).get_trace(*args, **kwargs)  # we need to feed the observed stuff into guide_trace

            for name in model_trace.observation_nodes:
                # this takes the dist and parameters and draws a sample to use as input to the guide
                # TODO: put this somewhere sensibler
                obs_dist = model_trace.nodes[name]["fn"]
                obs_args = model_trace.nodes[name]["args"]
                obs_kwargs = model_trace.nodes[name]["kwargs"]
                obs_val = obs_dist(*obs_args,
                                   **obs_kwargs)
                kwargs[name] = obs_val

            guide_trace = poutine.trace(poutine.replay(guide, model_trace)).get_trace(*args, **kwargs)
            # this is successfully setting the sampled calues to be those drawn from the model

            check_model_guide_match(model_trace, guide_trace)
            guide_trace = prune_subsample_sites(guide_trace)
            model_trace = prune_subsample_sites(model_trace)

            guide_trace.log_pdf()   # this needs to be here or guide_site["log_pdf"] gets a KeyError
            weight = 1.0 / self.num_particles
            yield weight, model_trace, guide_trace

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the loss (expectation over p of -log q)
        :rtype: float

        Performs backward on the loss. Num_particle many samples are used to form the estimators.
        """
        for weight, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            # can we just use loss = -guide_trace.log_pdf() ?

            loss = 0
            for name, guide_site in guide_trace.nodes.items():
                if guide_site["type"] == "sample":
                    loss -= guide_site["log_pdf"]

            # drop terms of weight zero to avoid nans
            if weight == 0.0:
                loss = torch_zeros_like(loss)

            loss *= weight

        # get gradients
        torch_backward(loss)

        loss = loss.data.numpy()[0]
        if np.isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
