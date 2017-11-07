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


class Model(anon.Latent):
    @anon.checked
    def __call__(self, xs):
        self.mu = Variable(torch.zeros(1))
        self.sigma = anon.param(Variable(torch.ones(1), requires_grad=True))
        self.ys = []
        self.xs = []
        for x in xs:
            prev = self.ys[-1] if self.ys else self.mu
            self.ys.append(anon.sample(dist.normal, prev, self.sigma))
            self.xs.append(anon.observe(dist.normal, x, self.ys[-1], self.sigma))


class Guide(anon.Latent):
    @anon.checked
    def __call__(self, xs):
        self.mus = [anon.param(Variable(torch.zeros(1), requires_grad=True)) for _ in range(len(xs))]
        self.sigmas = [anon.param(Variable(torch.ones(1), requires_grad=True)) for _ in range(len(xs))]
        self.ys = [None] * len(xs)
        for i in range(len(xs)):
            self.ys[i] = anon.sample(dist.normal, self.mus[i], self.sigmas[i])


def main(args):
    model = Model()
    guide = Guide()
    optim = Adam({"lr": 0.01})
    inference = SVI(model, guide, optim, loss="ELBO")
    data = Variable(torch.Tensor([0, 1, 1, 0, 1, 2]))
    for step in range(args.num_epochs):
        if step % 100 == 0:
            loss = inference.step(data)
            print('{}\t{:0.5g}'.format(step, loss))
    print('Parameters:')
    for name in pyro.get_param_store().get_all_param_names():
        print(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int)
    args = parser.parse_args()
    main(args)
