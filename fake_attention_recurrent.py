import theano
from theano import tensor

from blocks.bricks import (Brick, Initializable, Sequence,
                           Feedforward, Linear, Tanh)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Parallel, Distribute
from blocks.bricks.recurrent import recurrent, BaseRecurrent
from blocks.utils import dict_union, dict_subset, pack
from blocks.bricks.attention import (
    GenericSequenceAttention, AbstractAttentionRecurrent)
from match_functions import SumMatchFunction

class FakeAttentionRecurrent(AbstractAttentionRecurrent, Initializable):
    """Combines an attention mechanism and a recurrent transition.

    This brick equips a recurrent transition with an attention mechanism.
    In order to do this two more contexts are added: one to be attended and
    a mask for it. It is also possible to use the contexts of the given
    recurrent transition for these purposes and not add any new ones,
    see `add_context` parameter.

    At the beginning of each step attention mechanism produces glimpses;
    these glimpses together with the current states are used to compute the
    next state and finish the transition. In some cases glimpses from the
    previous steps are also necessary for the attention mechanism, e.g.
    in order to focus on an area close to the one from the previous step.
    This is also supported: such glimpses become states of the new
    transition.

    To let the user control the way glimpses are used, this brick also
    takes a "distribute" brick as parameter that distributes the
    information from glimpses across the sequential inputs of the wrapped
    recurrent transition.

    Parameters
    ----------
    transition : :class:`.BaseRecurrent`
        The recurrent transition.
    attention : :class:`.Brick`
        The attention mechanism.
    distribute : :class:`.Brick`, optional
        Distributes the information from glimpses across the input
        sequences of the transition. By default a :class:`.Distribute` is
        used, and those inputs containing the "mask" substring in their
        name are not affected.
    add_contexts : bool, optional
        If ``True``, new contexts for the attended and the attended mask
        are added to this transition, otherwise existing contexts of the
        wrapped transition are used. ``True`` by default.
    attended_name : str
        The name of the attended context. If ``None``, "attended"
        or the first context of the recurrent transition is used
        depending on the value of `add_contents` flag.
    attended_mask_name : str
        The name of the mask for the attended context. If ``None``,
        "attended_mask" or the second context of the recurrent transition
        is used depending on the value of `add_contents` flag.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    Wrapping your recurrent brick with this class makes all the
    states mandatory. If you feel this is a limitation for you, try
    to make it better! This restriction does not apply to sequences
    and contexts: those keep being as optional as they were for
    your brick.

    Those coming to Blocks from Groundhog might recognize that this is
    a `RecurrentLayerWithSearch`, but on steroids :)

    """
    def __init__(self, representationDim,transition, context_transition, distribute=None,
                 add_contexts=True,
                 attended_name=None, attended_mask_name=None,
                 **kwargs):
        super(FakeAttentionRecurrent, self).__init__(**kwargs)
        self._sequence_names = list(transition.apply.sequences)
        self._state_names = list(transition.apply.states)
        self._context_names = list(transition.apply.contexts)
        if add_contexts:
            if not attended_name:
                attended_name = 'attended_list'
            if not attended_mask_name:
                attended_mask_name = 'attended_mask_list'
            self._context_names += [attended_name, attended_mask_name]
        else:
            attended_name = self._context_names[0]
            attended_mask_name = self._context_names[1]
        if not distribute:
            normal_inputs = [name for name in self._sequence_names
                             if 'mask' not in name]
            distribute = Distribute(normal_inputs,
                                    'weighted_averages')
        self.representationDim=representationDim;
        self.transition = transition
        self.context_transition=context_transition
        self.distribute = distribute
        self.add_contexts = add_contexts
        self.attended_name = attended_name
        self.attended_mask_name = attended_mask_name

        self.preprocessed_attended_name = "preprocessed_" + self.attended_name

        self._glimpse_names = ['weighted_averages'] #unchanged
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_glimpses` signature.
        # self.previous_glimpses_needed = [
        #     name for name in self._glimpse_names
        #     if name in self.attention.take_glimpses.inputs]
        self.previous_glimpses_needed = []


        self.children = [self.transition, self.context_transition, self.distribute]

    def _push_allocation_config(self):
        self.distribute.source_dim = self.get_dim('states')
        self.distribute.target_dims = self.transition.get_dims(
            self.distribute.target_names)
    @application
    def take_glimpses(self, utterance_vector):
        r"""Compute glimpses with the attention mechanism.

        A thin wrapper over `self.attention.take_glimpses`: takes care
        of choosing and renaming the necessary arguments.

        Parameters
        ----------
        \*\*kwargs
            Must contain the attended, previous step states and glimpses.
            Can optionaly contain the attended mask and the preprocessed
            attended.

        Returns
        -------
        glimpses : list of :class:`~tensor.TensorVariable`
            Current step glimpses.

        """

        result = utterance_vector[3];

        # At this point kwargs may contain additional items.
        # e.g. AttentionRecurrent.transition.apply.contexts
        return result

    @take_glimpses.property('outputs')
    def take_glimpses_outputs(self):
        return self._glimpse_names

    @application
    def compute_states(self, **kwargs):
        r"""Compute current states when glimpses have already been computed.

        Combines an application of the `distribute` that alter the
        sequential inputs of the wrapped transition and an application of
        the wrapped transition. All unknown keyword arguments go to
        the wrapped transition.

        Parameters
        ----------
        \*\*kwargs
            Should contain everything what `self.transition` needs
            and in addition the current glimpses.

        Returns
        -------
        current_states : list of :class:`~tensor.TensorVariable`
            Current states computed by `self.transition`.

        """
        # make sure we are not popping the mask
        normal_inputs = [name for name in self._sequence_names
                         if 'mask' not in name]
        sequences = dict_subset(kwargs, normal_inputs, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        if self.add_contexts:
            kwargs.pop(self.attended_name)
            # attended_mask_name can be optional
            kwargs.pop(self.attended_mask_name, None)

        sequences.update(self.distribute.apply(
            as_dict=True, **dict_subset(dict_union(sequences, glimpses),
                                        self.distribute.apply.inputs)))
        current_states = self.transition.apply(
            iterate=False, as_list=True,
            **dict_union(sequences, kwargs))
        return current_states

    @compute_states.property('outputs')
    def compute_states_outputs(self):
        return self._state_names

    @application
    def transform_context(self,attended_list,attended_mask_list,states):
        preprocessed_attended_list=self.context_transition.preprocess(attended_list);
        utterance_attended=self.context_transition.apply(attended_list,preprocessed_attended_list,attended_mask_list, states, mask=tensor.ones([attended_list.shape[2],attended_list.shape[0]]));

        return utterance_attended;
    @recurrent
    def do_apply(self, **kwargs):
        r"""Process a sequence attending the attended context every step.

        In addition to the original sequence this method also requires
        its preprocessed version, the one computed by the `preprocess`
        method of the attention mechanism. Unknown keyword arguments
        are passed to the wrapped transition.

        Parameters
        ----------
        \*\*kwargs
            Should contain current inputs, previous step states, contexts,
            the preprocessed attended context, previous step glimpses.

        Returns
        -------
        outputs : list of :class:`~tensor.TensorVariable`
            The current step states and glimpses.

        """
        attended_list = kwargs[self.attended_name]
        preprocessed_attended_list = kwargs.pop(self.preprocessed_attended_name)
        attended_mask_list = kwargs.get(self.attended_mask_name)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        sequences = dict_subset(kwargs, self._sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self._state_names, pop=True)

        utterance_attended=self.context_transition.apply(attended_list,preprocessed_attended_list,attended_mask_list, states['states'], mask=tensor.ones([attended_list.shape[2],attended_list.shape[0]]));
        current_glimpses=self.take_glimpses(utterance_attended)#?
        current_states = self.compute_states(
            as_list=True,
            **dict_union(sequences, states, {'weighted_averages':current_glimpses}, kwargs))
        return current_states + [current_glimpses]


    @do_apply.property('sequences')
    def do_apply_sequences(self):
        return self._sequence_names

    @do_apply.property('contexts')
    def do_apply_contexts(self):
        return self._context_names + [self.preprocessed_attended_name]

    @do_apply.property('states')
    def do_apply_states(self):
        return self._state_names + self._glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self._state_names + self._glimpse_names

    @application
    def apply(self, **kwargs):
        """Preprocess a sequence attending the attended context at every step.

        Preprocesses the attended context and runs :meth:`do_apply`. See
        :meth:`do_apply` documentation for further information.

        """
        attended_list=kwargs[self.attended_name];
        preprocessed_attended_list=self.context_transition.preprocess(attended_list);
        return self.do_apply(
            **dict_union(kwargs,
                         {self.preprocessed_attended_name:
                          preprocessed_attended_list}))

    @apply.delegate
    def apply_delegate(self):
        # TODO: Nice interface for this trick?
        return self.do_apply.__get__(self, None)

    @apply.property('contexts')
    def apply_contexts(self):
        return self._context_names

    @application
    def initial_states(self, batch_size, **kwargs):
        return (pack(self.transition.initial_states(
                     batch_size, **kwargs)) +
                pack([tensor.zeros((batch_size, self.representationDim/2))]))

    @initial_states.property('outputs')
    def initial_states_outputs(self):
        return self.do_apply.states

    def get_dim(self, name):
        if name in self._glimpse_names:
            return self.representationDim/2
        #     return self.attention.get_dim(name)
        # if name == self.preprocessed_attended_name:
        #     (original_name,) = self.attention.preprocess.outputs
        #     return self.attention.get_dim(original_name)
        # if self.add_contexts:
        #     if name == self.attended_name:
        #         return self.attention.get_dim(
        #             self.attention.take_glimpses.inputs[0])
        #     if name == self.attended_mask_name:
        #         return 0
        return self.transition.get_dim(name)
