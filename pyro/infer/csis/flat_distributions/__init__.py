from __future__ import absolute_import, division, print_function

from pyro.infer.csis.flat_distributions.normal import FlatNormal
from pyro.distributions.random_primitive import RandomPrimitive

flat_normal = RandomPrimitive(FlatNormal)
