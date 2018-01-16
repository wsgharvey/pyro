from __future__ import absolute_import, division, print_function

# distribution classes
from pyro.distributions.uniform import UniformProposal

# function aliases
uniform_proposal = RandomPrimitive(UniformProposal)
