from theano import tensor

from blocks.bricks import Initializable, Random, Bias, NDimensionalSoftmax
from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import recurrent
from blocks.roles import add_role, COST
from blocks.utils import dict_union, dict_subset
from blocks.bricks.sequence_generators import (
    BaseSequenceGenerator)
#from attention import AttentionRecurrent
from attention_with_l import AttentionRecurrent_withL
from fake_attention_recurrent import FakeAttentionRecurrent


class SequenceGenerator(BaseSequenceGenerator):
    r"""A more user-friendly interface for :class:`BaseSequenceGenerator`.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component for the sequence generator.
    transition : instance of :class:`.BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : object, optional
        The attention mechanism to be added to ``transition``,
        an instance of
        :class:`~blocks.bricks.attention.AbstractAttention`.
    add_contexts : bool
        If ``True``, the
        :class:`.AttentionRecurrent` wrapping the
        `transition` will add additional contexts for the attended and its
        mask.
    \*\*kwargs : dict
        All keywords arguments are passed to the base class. If `fork`
        keyword argument is not provided, :class:`.Fork` is created
        that forks all transition sequential inputs without a "mask"
        substring in them.

    """
    def __init__(self, representation_dim,readout, transition, context_transition,attention=None,
                 use_step_decay_cost=False,
                 use_doubly_stochastic=False, lambda_ds=0.001,
                 use_concentration_cost=False, lambda_ct=10,
                 use_stablilizer=False, lambda_st=50,
                 add_contexts=True, **kwargs):
        self.use_doubly_stochastic = use_doubly_stochastic
        self.use_step_decay_cost = use_step_decay_cost
        self.use_concentration_cost = use_concentration_cost
        self.use_stablilizer = use_stablilizer
        self.lambda_ds = lambda_ds
        self.lambda_ct = lambda_ct
        self.lambda_st = lambda_st
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        self.context_transition=context_transition
        if attention:
            transition = AttentionRecurrent_withL(
                transition, context_transition,attention,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(representation_dim,transition,context_transition,
                                                name="with_fake_attention")

        super(SequenceGenerator, self).__init__(
            readout, transition, **kwargs)
        #self.children+=[self.context_transition]

    @application
    def cost_matrix(self, application_call, outputs, mask=None, **kwargs):
        """Returns generation costs for output sequences.

        See Also
        --------
        :meth:`cost` : Scalar cost.

        """
        # We assume the data has axes (time, batch, features, ...)
        batch_size = outputs.shape[1]

        # Prepare input for the iterative part
        states = dict_subset(kwargs, self._state_names, must_have=False)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        feedback = self.readout.feedback(outputs)
        inputs = self.fork.apply(feedback, as_dict=True)

        # Run the recurrent network
        results = self.transition.apply(
            mask=mask, return_initial_states=True, as_dict=True,
            **dict_union(inputs, states, contexts))

        # Separate the deliverables. The last states are discarded: they
        # are not used to predict any output symbol. The initial glimpses
        # are discarded because they are not used for prediction.
        # Remember, glimpses are computed _before_ output stage, states are
        # computed after.
        states = {name: results[name][:-1] for name in self._state_names}
        glimpses = {name: results[name][1:] for name in self._glimpse_names}

        # Compute the cost
        feedback = tensor.roll(feedback, 1, 0)
        feedback = tensor.set_subtensor(
            feedback[0],
            self.readout.feedback(self.readout.initial_outputs(batch_size)))
        readouts = self.readout.readout(
            feedback=feedback, **dict_union(states, glimpses, contexts))
        costs = self.readout.cost(readouts, outputs)

        if self.use_doubly_stochastic:
            # Doubly stochastic cost
            # \lambda\sum_{i}(1-\sum_{t}w_{t, i})^2
            # the first dimensions of weights returned by transition
            # is batch, time
            weights = glimpses['weights']
            weights_sum_time = tensor.sum(weights, 0)
            penalties = tensor.ones_like(weights_sum_time) - weights_sum_time
            penalties_squared = tensor.pow(penalties, 2)
            ds_costs = tensor.sum(penalties_squared, 1)
            costs += (self.lambda_ds * ds_costs)[None, :]

        def step_decay_cost(states):
            # shape is time, batch, features
            eta = 0.0001
            xi = 100
            states_norm = states.norm(2, axis=2)
            zz = tensor.zeros([1, states.shape[1]])
            padded_norm = tensor.join(0, zz, states_norm)[:-1, :]
            diffs = states_norm - padded_norm
            costs = eta * (xi ** diffs)
            return costs

        if self.use_step_decay_cost:
            costs += step_decay_cost(states['states'])

        def stablilizer_cost(states):
            states_norm = states.norm(2, axis=2)
            zz = tensor.zeros([1, states.shape[1]])
            padded_norm = tensor.join(0, zz, states_norm)[:-1, :]
            diffs = states_norm - padded_norm
            costs = tensor.pow(diffs, 2)
            return costs

        if self.use_stablilizer:
            costs += self.lambda_st * stablilizer_cost(states['states'])

        if self.use_concentration_cost:
            # weights has shape [batch, time, source sentence len]
            weights = glimpses['weights']
            maxis = tensor.max(weights, axis=2)
            lacks = tensor.ones_like(maxis) - maxis
            costs += self.lambda_ct * lacks

        if mask is not None:
            costs *= mask

        for name, variable in list(glimpses.items()) + list(states.items()):
            application_call.add_auxiliary_variable(
                variable.copy(), name=name)

        # This variables can be used to initialize the initial states of the
        # next batch using the last states of the current batch.
        for name in self._state_names:
            application_call.add_auxiliary_variable(
                results[name][-1].copy(), name=name+"_final_value")

        return costs

    @recurrent
    def generate(self, outputs, **kwargs):
        """A sequence generation step.

        Parameters
        ----------
        outputs : :class:`~tensor.TensorVariable`
            The outputs from the previous step.
        Notes
        -----
        The contexts, previous states and glimpses are expected as keyword
        arguments.

        """
        states = dict_subset(kwargs, self._state_names)
        # masks in context are optional (e.g. `attended_mask`)
        contexts = dict_subset(kwargs, self._context_names, must_have=False)
        glimpses = dict_subset(kwargs, self._glimpse_names)
        attended_list=contexts['attended_list'];
        attended_mask_list=contexts['attended_mask_list'];
        utterance_attended=self.transition.transform_context(attended_list,attended_mask_list,states['states']);
        next_glimpses=self.transition.take_glimpses(
            as_dict=True,
            **dict_union(
                states,
                {'attended_list': utterance_attended,
                 'attended_mask_list': tensor.ones([attended_list.shape[0],attended_list.shape[2]])}))
        next_readouts = self.readout.readout(
            feedback=self.readout.feedback(outputs),
            **dict_union(states, next_glimpses, contexts))
        next_outputs = self.readout.emit(next_readouts)
        next_costs = self.readout.cost(next_readouts, next_outputs)
        next_feedback = self.readout.feedback(next_outputs)
        next_inputs = (self.fork.apply(next_feedback, as_dict=True)
                       if self.fork else {'feedback': next_feedback})
        next_states = self.transition.compute_states(
            as_list=True,
            **dict_union(next_inputs, states, next_glimpses, contexts))
        return (next_states + [next_outputs] +
                list(next_glimpses.values()) + [next_costs])

    # @recurrent
    # def generate_noWA(self, outputs, **kwargs):
    #     """A sequence generation step.
    #
    #     Parameters
    #     ----------
    #     outputs : :class:`~tensor.TensorVariable`
    #         The outputs from the previous step.
    #     Notes
    #     -----
    #     The contexts, previous states and glimpses are expected as keyword
    #     arguments.
    #
    #     """
    #     states = dict_subset(kwargs, self._state_names)
    #     # masks in context are optional (e.g. `attended_mask`)
    #     contexts = dict_subset(kwargs, self._context_names, must_have=False)
    #     glimpses = dict_subset(kwargs, self._glimpse_names)
    #     attended_list=contexts['attended_list'];
    #     attended_mask_list=contexts['attended_mask_list'];
    #     utterance_attended=self.transition.transform_context(attended_list,attended_mask_list,states['states']);
    #     next_glimpses_tensor=self.transition.take_glimpses(utterance_attended);
    #     next_glimpses={'weighted_averages':next_glimpses_tensor};
    #     next_readouts = self.readout.readout(
    #         feedback=self.readout.feedback(outputs),
    #         **dict_union(states, next_glimpses, contexts))
    #     next_outputs = self.readout.emit(next_readouts)
    #     next_costs = self.readout.cost(next_readouts, next_outputs)
    #     next_feedback = self.readout.feedback(next_outputs)
    #     next_inputs = (self.fork.apply(next_feedback, as_dict=True)
    #                    if self.fork else {'feedback': next_feedback})
    #     next_states = self.transition.compute_states(
    #         as_list=True,
    #         **dict_union(next_inputs, states, next_glimpses, contexts))
    #     return (next_states + [next_outputs] +
    #             list(next_glimpses.values()) + [next_costs])


    @generate.delegate
    def generate_delegate(self):
        return self.transition.apply

    @generate.property('states')
    def generate_states(self):
        return self._state_names + ['outputs'] + self._glimpse_names

    @generate.property('outputs')
    def generate_outputs(self):
        return (self._state_names + ['outputs'] +
                self._glimpse_names + ['costs'])

    def get_dim(self, name):
        if name in (self._state_names + self._context_names +
                    self._glimpse_names):
            return self.transition.get_dim(name)
        elif name == 'outputs':
            return self.readout.get_dim(name)
        return super(BaseSequenceGenerator, self).get_dim(name)

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        # TODO: support dict of outputs for application methods
        # to simplify this code.
        state_dict = dict(
            self.transition.initial_states(
                batch_size, as_dict=True, *args, **kwargs),
            outputs=self.readout.initial_outputs(batch_size))
        return [state_dict[state_name]
                for state_name in self.generate.states]

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.generate.states
