from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution

from pyro.distributions.uniform import Uniform
from pyro.distributions.beta import Beta


class UniformProposal(Uniform):
    """
    Proposal for a uniform distribution over the continuous interval `[a, b]`,
    with a peak with specified mode and certainty.

    Taken from:
    https://github.com/probprog/anglican-infcomp/blob/master/src/anglican/infcomp/dists.clj

    :param torch.autograd.Variable a: lower bound (real).
    :param torch.autograd.Variable b: upper bound (real).
    :param torch.autograd.Variable mode: mode of distribution (real, in [a, b])
    :param torch.autograd.Variable certainty: certainty (real, in [0, inf]).
        Should be greater than `a`.
    """
    reparameterized = False  # XXX Why is this marked non-differentiable?

    def __init__(self, a, b, mode, certainty, batch_size=None, *args, **kwargs):
        if a.size() != b.size():
            raise ValueError("Expected a.size() == b.size(), but got {} vs {}".format(a.size(), b.size()))
        if a.size() != mode.size():
            raise ValueError("Expected a.size() == mode.size(), but got {} vs {}".format(a.size(), mode.size()))
        if a.size() != certainty.size():
            raise ValueError("Expected a.size() == certainty.size(), but got {} vs {}".format(a.size(), certainty.size()))
        normalised_mode = (mode-a) / (b-a)
        normalised_certainty = certainty + 2
        self.beta = Beta(normalised_mode * (normalised_certainty - 2),
                         (1 - normalised_mode) * (normalised_certainty - 2))
        super(UniformProposal, self).__init__(a, b, batch_size, *args, **kwargs)

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        eps = self.beta.sample()
        return self.a + torch.mul(eps, self.b - self.a)

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        uniform_pdf = super(UniformProposal, self).batch_log_pdf(x)
        normalised_beta_pdf = self.beta.batch_log_pdf((x-self.a)/(self.b-self.a))
        return normalised_beta_pdf + uniform_pdf

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        # TODO: this should be fairly easy to implement using self.beta
        raise NotImplementedError

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        # TODO: this should be fairly easy to implement using self.beta
        raise NotImplementedError
