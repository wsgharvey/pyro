from __future__ import absolute_import, division, print_function

# distribution classes
from pyro.distributions.random_primitive import RandomPrimitive
from pyro.infer.csis.proposal_dists.uniform import UniformProposal

# function aliases
uniform_proposal = RandomPrimitive(UniformProposal)
