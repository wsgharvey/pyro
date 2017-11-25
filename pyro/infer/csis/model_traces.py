from __future__ import absolute_import, division, print_function

import pyro
import pyro.poutine as poutine


def get_trace_from_prior(model, guide, *args, **kwargs):

    model_trace = poutine.trace(model).get_trace(*args, **kwargs)

    for name in model_trace.observation_nodes:
        # replace the value of each observe with the sampled value
        obs_dist = model_trace.nodes[name]["fn"]
        obs_args = model_trace.nodes[name]["args"]
        obs_kwargs = model_trace.nodes[name]["kwargs"]
        model_trace.nodes[name]["value"] = obs_dist(*obs_args,
                                                    **obs_kwargs)
    return model_trace
