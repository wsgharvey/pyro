from __future__ import absolute_import, division, print_function

import pyro
import pyro.poutine


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
        for _ in range(self.num_particles):

            # take a trace from the model and then take a trace from the guide with the samples constrained to be the same
            model_trance = poutine.trace(model).get_trace(*args, **kwargs)
            guide_trace = poutine.trace(poutine.replay(guide, model_trace).get_trace(*args, **kwargs))

            check_model_guide_match(model_trace, guide_trace)
            guide_trace = prune_subsample_sites(guide_trace)
            model_trace = prune_subsample_sites(model_trace)

            log_r = model_trace.log_pdf() - guide_trace.log_pdf()
            weight = 1.0 / self.num_particles
            yield weight, model_trace, guide_trace, log_r

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the loss (expectation over p of -log q)
        :rtype: float

        Performs backward on the loss. Num_particle many samples are used to form the estimators.
        """
        for weight, model_trace, guide_trace, log_r in selg._get_traces(model, guide, *args, **kwargs)
