from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable

import pyro
import pyro.anon as anon
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam

# ys[0] --> ys[1] --> ... --> ys[n]
#   |         |                 |
#   V         V                 V
# xs[0]     xs[1]             xs[n]


@anon.function
def model(latent, xs):
    latent.mu = Variable(torch.zeros(1))
    latent.sigma = anon.param(Variable(torch.ones(1), requires_grad=True))
    latent.ys = []
    latent.xs = []
    for x in xs:
        prev = latent.ys[-1] if latent.ys else latent.mu
        latent.ys.append(anon.sample(dist.normal, prev, latent.sigma))
        latent.xs.append(anon.observe(dist.normal, x, latent.ys[-1], latent.sigma))


@anon.function
def guide(latent, xs):
    latent.mus = [anon.param(Variable(torch.zeros(1), requires_grad=True)) for _ in range(len(xs))]
    latent.sigmas = [anon.param(Variable(torch.ones(1), requires_grad=True)) for _ in range(len(xs))]
    latent.ys = [None] * len(xs)
    for i in range(len(xs)):
        latent.ys[i] = anon.sample(dist.normal, latent.mus[i], latent.sigmas[i])


def main(args):
    optim = Adam({"lr": 0.1})
    inference = SVI(model, guide, optim, loss="ELBO")
    data = Variable(torch.Tensor([0, 1, 2, 4, 4, 4]))

    print('Step\tLoss')
    for step in range(args.num_epochs):
        if step % 100 == 0:
            loss = inference.step(data)
            print('{}\t{:0.5g}'.format(step, loss))

    print('Parameters:')
    for name in sorted(pyro.get_param_store().get_all_param_names()):
        print('{} = {}'.format(name, pyro.param(name).data.numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    args = parser.parse_args()
    main(args)
