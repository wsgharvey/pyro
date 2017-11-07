from __future__ import absolute_import, division, print_function

import functools
import numbers

import numpy as np
import torch

import pyro

# Optional safety devices.
_CHECKING = []
_PYRO_PENDING = set()  # Set of sample and param objects.
_PYRO_DONE = set()  # Set of addresses.


def checked(fn):
    """
    Decorator for model and guide functions to check for safety.
    """

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        if _CHECKING:
            raise RuntimeError('@checked functions do not support recursion')
        _CHECKING.append(None)
        _PYRO_PENDING.clear()
        _PYRO_DONE.clear()
        try:
            result = fn(*args, **kwargs)
        finally:
            _CHECKING.pop()
        if _PYRO_PENDING:
            raise RuntimeError('\n'.join(['Unassigned sites:'] + list(map(str, _PYRO_PENDING))))
        return result

    return decorated


# Deferred sample site, will not run until stored in a _Latent.
class sample(object):
    __doc__ = pyro.sample.__doc__

    def __init__(self, fn, *args, **kwargs):
        if _CHECKING:
            _PYRO_PENDING.add(self)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.name = None

    def bind(self, name):
        if _CHECKING:
            _PYRO_PENDING.remove(self)
        self.name = name
        return pyro.sample(name, self.fn, *self.args, **self.kwargs)


# Deferred observe site, will not run until stored in a _Latent.
@functools.wraps(pyro.observe)
def observe(fn, obs, *args, **kwargs):
    kwargs["obs"] = obs
    return sample(fn, *args, **kwargs)


# Deferred param site, will not run until stored in a _Latent.
class param(object):
    __doc__ = pyro.sample.__doc__

    def __init__(self, *args, **kwargs):
        if _CHECKING:
            _PYRO_PENDING.add(self)
        self.args = args
        self.kwargs = kwargs
        self.name = None

    def bind(self, name):
        if _CHECKING:
            _PYRO_PENDING.remove(self)
        self.name = name
        return pyro.param(name, *self.args, **self.kwargs)


class _Latent(object):
    """
    Base class for latent state containers.
    """
    def __init__(self, address='latent'):
        super(_Latent, self).__setattr__('_address', address)

    def set_address(self, address):
        assert self._address == 'latent'
        super(_Latent, self).__setattr__('_address', address)


# This attempts to disallow a _Latent from storing an unregistered container
# that might accidentally hold unbound anon.sample or anon.param objects. If
# this ends up being too restrictive, we might drop this check.
_ALLOWED_TYPES = (
    type(None),
    str,
    numbers.Number,
    torch.Tensor,
    torch.autograd.Variable,
    np.ndarray,
)


class Latent(_Latent):
    """
    Object to hold latent state.
    """

    def __setattr__(self, name, value):
        address = '{}.{}'.format(self._address, name)
        if _CHECKING:
            if address in _PYRO_DONE:
                raise RuntimeError('Cannot overwrite {}'.format(address))
        if isinstance(value, _Latent):
            value.set_address(address)
        elif isinstance(value, (sample, param)):
            value = value.bind(address)
            if _CHECKING:
                _PYRO_DONE.add(address)
        elif isinstance(value, list):
            value = LatentList(address, value)
        elif isinstance(value, dict):
            value = LatentDict(address, value)
        elif not isinstance(value, _ALLOWED_TYPES):
            raise TypeError('Latent cannot store objects of type {}'.format(type(value)))
        super(Latent, self).__setattr__(name, value)

    # TODO Make mutation methods safe.


class LatentDict(_Latent, dict):
    """
    Dict-like object to hold latent state.
    """

    def __init__(self, address='latent', raw_value=None):
        super(LatentDict, self).__init__(address)
        if raw_value is not None:
            assert not isinstance(raw_value, _Latent)
            for key, value in raw_value:
                self[key] = value

    def __setitem__(self, key, value):
        address = '{}[{:r}]'.format(self._address, key)
        if _CHECKING:
            if address in _PYRO_DONE:
                raise RuntimeError('Cannot overwrite {}'.format(address))
        if isinstance(value, _Latent):
            value.set_address(address)
        elif isinstance(value, (sample, param)):
            value = value.bind(address)
            if _CHECKING:
                _PYRO_DONE.add(key)
        elif isinstance(value, list):
            value = LatentList(address, value)
        elif isinstance(value, dict):
            value = LatentDict(address, value)
        elif not isinstance(value, _ALLOWED_TYPES):
            raise TypeError('LatentDict cannot store objects of type {}'.format(type(value)))
        super(LatentDict, self).__setitem__(key, value)

    def setdefault(self, key, value):
        try:
            return self[key]
        except KeyError:
            self[key] = value
            return self[key]

    # TODO Make mutation methods safe.


class LatentList(_Latent, list):
    """
    List-like object to hold latent state.
    """

    def __init__(self, address='latent', raw_value=None):
        super(LatentList, self).__init__(address)
        if raw_value is not None:
            assert not isinstance(raw_value, _Latent)
            for item in raw_value:
                self.append(item)

    def __setitem__(self, pos, value):
        address = '{}[{}]'.format(self._address, pos)
        if _CHECKING:
            if address in _PYRO_DONE:
                raise RuntimeError('Cannot overwrite {}'.format(address))
        if isinstance(value, _Latent):
            value.set_address(address)
        elif isinstance(value, (sample, param)):
            value = value.bind(address)
            if _CHECKING:
                _PYRO_DONE.add(address)
        elif isinstance(value, list):
            value = LatentList(address, value)
        elif isinstance(value, dict):
            value = LatentDict(address, value)
        elif not isinstance(value, _ALLOWED_TYPES):
            raise TypeError('LatentList cannot store objects of type {}'.format(type(value)))
        super(LatentList, self).__setitem__(pos, value)

    def append(self, value):
        pos = len(self)
        super(LatentList, self).append(None)
        self[pos] = value

    # TODO Make mutation methods safe.
