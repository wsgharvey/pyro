from __future__ import absolute_import, division, print_function

import pyro

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
