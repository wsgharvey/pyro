from __future__ import absolute_import, division, print_function

import pyro


class sample(object):
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.name = None

    def bind(self, name):
        self.name = name
        return pyro.sample(self, self.fn, *self.args, **self.kwargs)


def observe(fn, obs, *args, **kwargs):
    kwargs["obs"] = obs
    return sample(fn, *args, **kwargs)


class param(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.name = None

    def bind(self, name):
        self.name = name
        return pyro.param(self, *self.args, **self.kwargs)


class Latent(object):
    def __init__(self, address=''):
        self._address = address

    def set_address(self, address):
        assert self._address == ''
        self._address = address


class LatentObject(Latent):
    def __setattr__(self, name, value):
        address = '{}.{}'.format(self._address, name)
        if isinstance(value, Latent):
            value.set_address(address)
        elif isinstance(value, (sample, param)):
            value = value.bind(address)
        elif isinstance(value, list):
            value = LatentList(address, value)
        elif isinstance(value, dict):
            value = LatentDict(address, value)
        super(LatentObject, self).__setattr__(name, value)


class LatentDict(dict, Latent):
    def __init__(self, address='', raw_value=None):
        super(LatentDict, self).__init__(address)
        if raw_value is not None:
            assert not isinstance(raw_value, Latent)
            for key, value in raw_value:
                self[key] = value

    def __setitem__(self, key, value):
        address = '{}[{}]'.format(self._address, key)
        if isinstance(value, Latent):
            value.set_address(address)
        elif isinstance(value, (sample, param)):
            value = value.bind(address)
        elif isinstance(value, list):
            value = LatentList(address, value)
        elif isinstance(value, dict):
            value = LatentDict(address, value)
        super(LatentDict, self).__setitem__(key, value)

    def setdefault(self, key, value):
        try:
            return self[key]
        except KeyError:
            self[key] = value
            return self[key]


class LatentList(list, Latent):
    def __init__(self, address='', raw_value=None):
        super(LatentList, self).__init__(address)
        if raw_value is not None:
            assert not isinstance(raw_value, Latent)
            for item in raw_value:
                self.append(item)

    def __setitem__(self, pos, value):
        address = '{}[{}]'.format(self._address, pos)
        if isinstance(value, Latent):
            value.set_address(address)
        elif isinstance(value, (sample, param)):
            value = value.bind(address)
        elif isinstance(value, list):
            value = LatentList(address, value)
        elif isinstance(value, dict):
            value = LatentDict(address, value)
        super(LatentDict, self).__setitem__(pos, value)

    def append(self, value):
        pos = len(self)
        super(LatentList, self).append(None)
        self[pos] = value
