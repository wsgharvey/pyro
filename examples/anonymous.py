from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable

import pyro.anon as anon
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam

# ys[0] --> ys[1] --> ... --> ys[n]
#   |         |                 |
#   V         V                 V
# xs[0]     xs[1]             xs[n]


class Model(anon.Latent):
    def __call__(self, xs):
        self.mu = anon.param(Variable(torch.zeros(1)))
        self.sigma = anon.param(Variable(torch.ones(1)))
        self.ys = []
        self.xs = []
        for x in xs:
            prev = self.ys[-1] if self.ys else self.mu
            self.ys.append(anon.sample(dist.normal, prev, self.sigma))
            self.xs.append(anon.observe(dist.normal, x, self.ys[-1], self.sigma))


class Guide(anon.Latent):
    def __call__(self, xs):
        self.mus = [anon.param(Variable(torch.zeros(1))) for _ in range(len(xs))]
        self.sigmas = [anon.param(Variable(torch.ones(1))) for _ in range(len(xs))]
        self.ys = [None] * len(xs)
        for i in range(len(xs)):
            self.ys[i] = anon.sample(dist.normal, self.mus[i], self.sigmas[i])


def main(args):
    model = Model()
    guide = Guide()
    optim = Adam({"lr": 0.01})
    inference = SVI(model, guide, optim, loss="ELBO")
    data = Variable(torch.Tensor([0, 1, 1, 0, 1, 2]))
    for _ in range(args.num_epochs):
        inference.step(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    args = parser.parse_args()
    main(args)
