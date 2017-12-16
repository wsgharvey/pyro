from __future__ import absolute_import, division, print_function

from pyro.distributions.normal import Normal


class FlatNormal(Normal):
    """
    proposal for normal with wider tails to better represent uncommon sample
    values
    """
