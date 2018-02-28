from __future__ import absolute_import, division, print_function

import pyro
from pyro.infer.importance import Importance
from pyro.infer.csis.util import sample_from_prior
from pyro.infer.csis.loss import Loss

import torch


class CSIS(Importance):
    def __init__(self,
                 model,
                 guide,
                 num_samples):
        """
        Constructor
        """
        super(CSIS, self).__init__(model, guide, num_samples)
        self.model_args_set = False
        self.compiler_args_set = False
        self.compiler_initiated = False

    def set_model_args(self, *args, **kwargs):
        """
        must be called before running `compile` (even with no arguments)
        set the arguments to be used when compiling the model
        """
        self.model_args = args
        self.model_kwargs = kwargs
        self.model_args_set = True

    def set_compiler_args(self,
                          valid_size=10,
                          valid_frequency=10,
                          num_particles=10):
        """
        set the compiler properties - if not called before `compile`, defaults
        will be used
        """
        self.valid_size = valid_size
        self.valid_frequency = valid_frequency
        self.num_particles = num_particles
        self.compiler_args_set = True

    def _init_compiler(self):
        """
        internal function to create validation batch etc.
        """
        if not self.model_args_set:
            raise ValueError("Must set model arguments before compiling")

        if not self.compiler_args_set:
            self.set_compiler_args()

        self.valid_batch = [sample_from_prior(self.model,
                                              *self.model_args,
                                              **self.model_kwargs)
                            for _ in range(self.valid_size)]
        self.iterations = 0
        self.training_losses = []
        self.valid_losses = []
        self.compiler_initiated = True

    def compile(self,
                num_steps,
                optim=None,
                cuda=False):
        """
        :num_steps:     Number of iterations to perform
        :optim:         Torch optimiser object - if None, will use most recent

        :returns: None
        Does some training steps
        """
        if not self.compiler_initiated:
            self._init_compiler()

        if optim is not None:
            self.optim = optim

        loss = Loss(self.model,
                    self.guide,
                    self.model_args,
                    self.model_kwargs,
                    self.num_particles,
                    cuda)

        for _step in range(num_steps):
            self.optim.zero_grad()
            training_loss = loss.loss(self.model,
                                      self.guide,
                                      grads=True)
            self.optim.step()
            print("LOSS: {}".format(training_loss))
            self.training_losses.append(training_loss)
            self.iterations += 1
            if self.iterations % self.valid_frequency == 0:
                valid_loss = loss.loss(self.model,
                                       self.guide,
                                       grads=False,
                                       batch=self.valid_batch)
                self.valid_losses.append((self.iterations, valid_loss))
                print("                                     VALIDATION LOSS IS {}".format(valid_loss))

    def get_compile_log(self):
        """
        returns object with information about losses etc.
        """
        return {"validation": self.valid_losses,
                "training": list(enumerate(self.training_losses))}

    def get_last_optim(self):
        return self.optim

    def sample_from_prior(self):
        """
        returns a trace sampled from the prior, without conditioning
        - values at observe statements are also randomly sampled from the
          distribution
        """
        return sample_from_prior(self.model,
                                 *self.model_args,
                                 **self.model_kwargs)
