from __future__ import absolute_import, division, print_function

import pyro
from pyro.infer.importance import Importance
from pyro.infer.csis.util import sample_from_prior
from pyro.infer.csis.loss import Loss

import torch


class CSIS(Importance):
    """
    Compiled Sequential Importance Sampling, allowing compilation of a guide
    program to minimise KL(model posterior || guide)

    **Reference**
    "Inference Compilation and Universal Probabilistic Programming" `pdf https://arxiv.org/pdf/1610.09900.pdf`

    :param model: probabilistic model defined as a function
    :param guide: guide defined as a torch.nn.Module. guide.forward must be a
        function which samples from the approximate posterior. Must accept
        keyword arguments with names of observed sites in model.
    :param int num_samples: The number of importance-weighted samples to draw
    """
    def __init__(self,
                 model,
                 guide,
                 num_samples):
        super(CSIS, self).__init__(model, guide, num_samples)
        self.model_args_set = False
        self.compiler_args_set = False
        self.compiler_initiated = False

    def set_model_args(self, *args, **kwargs):
        """
        Sets model arguments to use during compilation. All arguments received
        are passed directly to the model every time it is run.

        Must be called before running `compile` (even with no arguments).
        """
        self.model_args = args
        self.model_kwargs = kwargs
        self.model_args_set = True

    def set_compiler_args(self,
                          validation_size=10,
                          valid_frequency=10,
                          num_particles=10):
        """
        Set hyperparameters for compilation. If not called before `compile`,
        default values will be used.

        :param int validation_size: Size of validation batch to use.
        :param int validation_frequency: Number of steps to run between
            validations.
        :param int num_particles: Number of particles to use.
        """
        self.valid_size = validation_size
        self.valid_frequency = valid_frequency
        self.num_particles = num_particles
        self.compiler_args_set = True

    def _init_compiler(self):
        """
        Internal function to create validation batch etc.
        """
        if not self.model_args_set:
            raise ValueError("Must set model arguments before compiling.")

        if not self.compiler_args_set:
            self.set_compiler_args()

        self.valid_batch = [sample_from_prior(self.model,
                                              *self.model_args,
                                              **self.model_kwargs)
                            for _ in range(self.valid_size)]
        self.iterations = 0
        self.training_losses = []
        self.valid_losses = []
        self.loss = Loss(self.model,
                         self.guide,
                         self.model_args,
                         self.model_kwargs,
                         self.num_particles,
                         cuda)
        self.compiler_initiated = True

    def batch_loss(self,
                   grads=True,
                   cuda=False):
        """
        Calculate a loss and (optionally) perform `torch.backward` to
        calculate gradients.

        :param grads bool:    Whether to calculate gradients of the loss.
        :param cuda bool:     Whether to use CUDA
        """
        training_loss = self.loss.loss(self.model,
                                       self.guide,
                                       grads=True)
        return training_loss

    def compile(self,
                num_steps,
                optim=None,
                cuda=False):
        """
        :param num_steps int: Number of iterations to perform
        :param optim:         Torch optimiser object - if None, will use most recent
        :param cuda bool:     Whether to use CUDA

        Performs training steps using arguments specified by `set_model_args`
        and `set_compiler_args`.
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
        Returns object with information about losses etc.
        """
        return {"validation": self.valid_losses,
                "training": list(enumerate(self.training_losses))}

    def get_last_optim(self):
        """
        Returns last optimiser used.

        Used to find state of optimiser.
        """
        return self.optim

    def sample_from_prior(self):
        """
        Returns a trace sampled from the model without conditioning. Values at
        observe statements are also randomly sampled from the distribution.
        """
        return sample_from_prior(self.model,
                                 *self.model_args,
                                 **self.model_kwargs)
