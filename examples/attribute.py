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
    latent.sigma.param(Variable(torch.ones(1), requires_grad=True))
    latent.ys = [object() for _ in xs]
    latent.xs = [object() for _ in xs]
    y = latent.mu
    for i, x in enumerate(xs):
        y = latent.ys[i].sample(dist.normal, y, latent.sigma)
        latent.xs[i].observe(dist.normal, x, y, latent.sigma)


@anon.function
def guide(latent, xs):
    latent.mus = [object() for _ in xs]
    latent.sigmas = [object() for _ in xs]
    for i in range(len(xs)):
        latent.mus[i].param(Variable(torch.zeros(1), requires_grad=True))
        latent.sigmas[i].param(Variable(torch.ones(1), requires_grad=True))
    latent.ys = [object() for _ in xs]
    for y, mu, sigma in zip(latent.ys, latent.mus, latent.sigmas):
        y.sample(dist.normal, mu, sigma)


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
