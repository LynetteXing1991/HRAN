from theano import tensor
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks import (Brick, Initializable, Sequence,
                           Feedforward, Linear, Tanh)
from blocks.utils import dict_union, dict_subset, pack


class ShallowEnergyComputer(Initializable, Feedforward):
    """A simple energy computer: first tanh, then weighted sum."""
    @lazy()
    def __init__(self, **kwargs):
        super(ShallowEnergyComputer, self).__init__(**kwargs)
        self.tanh = Tanh()
        self.linear = Linear(use_bias=False)
        self.children = [self.tanh, self.linear]

    @application
    def apply(self, *args):
        output = args
        output = self.tanh.apply(*pack(output))
        output = self.linear.apply(*pack(output))
        return output

    @property
    def input_dim(self):
        return self.children[1].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[1].input_dim = value

    @property
    def output_dim(self):
        return self.children[1].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[1].output_dim = value


class SumMatchFunction(Initializable, Feedforward):

    @lazy()
    def __init__(self, **kwargs):
        super(SumMatchFunction, self).__init__(**kwargs)
        self.shallow = ShallowEnergyComputer()
        self.children = [self.shallow]

    @application
    def apply(self, states, attended):
        match_vectors = states + attended
        energies = self.shallow.apply(*pack(match_vectors))
        energies = energies.reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @property
    def input_dim(self):
        return self.children[0].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[0].input_dim = value

    @property
    def output_dim(self):
        return self.children[0].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[0].output_dim = value

class SumMatchFunction_posTag(Initializable, Feedforward):

    @lazy()
    def __init__(self, **kwargs):
        super(SumMatchFunction_posTag, self).__init__(**kwargs)
        self.shallow = ShallowEnergyComputer()
        self.children = [self.shallow]

    @application
    def apply(self, states, attended,posTag):
        match_vectors = states + attended+posTag
        energies = self.shallow.apply(*pack(match_vectors))
        energies = energies.reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @property
    def input_dim(self):
        return self.children[0].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[0].input_dim = value

    @property
    def output_dim(self):
        return self.children[0].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[0].output_dim = value


class CatMatchFunction(Initializable, Feedforward):

    @lazy()
    def __init__(self, **kwargs):
        super(CatMatchFunction, self).__init__(**kwargs)
        self.shallow = ShallowEnergyComputer()
        self.children = [self.shallow]

    @application
    def apply(self, states, attended):
        states = tensor.repeat(states[None, :, :], attended.shape[0], axis=0)
        match_vectors = tensor.concatenate([states, attended], axis=2)
        energies = self.shallow.apply(*pack(match_vectors))
        energies = energies.reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @property
    def input_dim(self):
        return self.children[0].input_dim

    @input_dim.setter
    def input_dim(self, value):
        # because we concat to input_dim is match_dim * 2
        self.children[0].input_dim = value * 2

    @property
    def output_dim(self):
        return self.children[0].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[0].output_dim = value


class DotMatchFunction(Initializable, Feedforward):

    @lazy()
    def __init__(self, **kwargs):
        super(DotMatchFunction, self).__init__(**kwargs)

    @application
    def apply(self, states, attended):
        match_vectors = tensor.tensordot(
            attended, states, axes=[2, 1])[:, 0, :]
        energies = tensor.exp(match_vectors)
        return energies


class GeneralMatchFunction(Initializable, Feedforward):

    @lazy()
    def __init__(self, **kwargs):
        super(GeneralMatchFunction, self).__init__(**kwargs)
        self.linear = Linear(use_bias=False)
        self.children = [self.linear]

    @application
    def apply(self, states, attended):
        states = self.linear.apply(*pack(states))
        match_vectors = tensor.tensordot(
            attended, states, axes=[2, 1])[:, 0, :]
        energies = tensor.exp(match_vectors)
        return energies

    @property
    def input_dim(self):
        return self.children[0].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[0].input_dim = value
        self.children[0].output_dim = value
