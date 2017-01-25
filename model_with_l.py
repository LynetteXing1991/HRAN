from theano import tensor
from toolz import merge

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
# from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter)
from blocks.roles import add_role, WEIGHT
from blocks.utils import shared_floatx_nans

from picklable_itertools.extras import equizip
from attention_with_posTagging import SequenceContentAttention_withExInput
from attention import SequenceContentAttention
from SequenceGenerator import SequenceGenerator
from GRU import GRU
from blocks.bricks.recurrent import recurrent
from blocks.utils import dict_union
from match_functions import (
    SumMatchFunction, CatMatchFunction,
    DotMatchFunction, GeneralMatchFunction,SumMatchFunction_posTag)


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class LookupFeedbackWMT15(LookupFeedback):
    """Zero-out initial readout feedback by checking its value."""

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in range(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup


class BidirectionalWMT15(Bidirectional):
    """Wrap two Gated Recurrents each having separate parameters."""

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]


class BidirectionalEncoder(Initializable):
    """Encoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        self.lookup = LookupTable(name='embeddings')
        self.bidir = BidirectionalWMT15(
            GatedRecurrent(activation=Tanh(), dim=state_dim))
        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.lookup, self.bidir,
                         self.fwd_fork, self.back_fork]

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.bidir.children[0].get_dim(name)
                                     for name in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.bidir.children[1].get_dim(name)
                                      for name in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation'])
    def apply(self, source_sentence, source_sentence_mask):
        # Time as first dimension
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        embeddings = self.lookup.apply(source_sentence)

        representation = self.bidir.apply(
            merge(self.fwd_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask})
        )
        return representation

class SentenceEncoder(Initializable):
    """Encoder of RNNsearch model."""

    def __init__(self, embedding_dim, state_dim, use_local_attention=False, window_size=10,**kwargs):
        super(SentenceEncoder, self).__init__(**kwargs)

        self.embedding_dim=embedding_dim
        self.state_dim = state_dim
        self.rnn=GRU(activation=Tanh(), dim=state_dim,attended_dim=embedding_dim);
        self.input_fork=Fork(
            [name for name in self.rnn.apply.sequences
             if name != 'mask'], prototype=Linear(), name='input_fork')
        self.energy_computer = SumMatchFunction_posTag(name="wordAtt_energy_comp")
        self.attention = SequenceContentAttention_withExInput(
            state_names=['states'],
            state_dims=[state_dim],
            attended_dim=embedding_dim,
            match_dim=state_dim,
            posTag_dim=self.state_dim,
            energy_computer=self.energy_computer,
            use_local_attention=use_local_attention,
            window_size=window_size,
            name="word_attention")

        self.children = [self.rnn,self.input_fork,self.attention]

    def _push_allocation_config(self):

        self.input_fork.input_dim = self.embedding_dim
        self.input_fork.output_dims = [self.rnn.get_dim(name)
                                     for name in self.input_fork.output_names]
        self.attention.state_dims=[self.state_dim]
        self.attention.state_dim=self.state_dim

    @recurrent(sequences=['attended', 'preprocessed_attended','attended_mask','mask'],
               states=['states'], outputs=['states'], contexts=['decoder_states'])
    def do_apply(self,attended,preprocessed_attended,attended_mask, decoder_states, states,mask=None):
        current_glimpses = self.attention.take_glimpses(
            attended,
            states,
            preprocessed_attended,
            attended_mask,
            states,
            **{'states':decoder_states});
        inputs=merge(self.input_fork.apply(current_glimpses[0], as_dict=True),{'states':states});

        next_states=self.rnn.apply(iterate=False,**inputs)
        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) *states)
        return next_states


    @application(inputs=['attended', 'preprocessed_attended','attended_mask','decoder_states','mask'],
                 outputs=['cxt_representation'])
    def apply(self, attended,preprocessed_attended,attended_mask, decoder_states, mask=None):
        # Time as first dimension
        mask = mask.T
        cxt_representation=self.do_apply(attended,preprocessed_attended,attended_mask, decoder_states, mask=mask)

        return cxt_representation

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in [ 'states']:
            return self.state_dim
    @application(inputs=['attended'], outputs=['preprocessed_attended'])
    def preprocess(self, attended):
        """Preprocess the sequence for computing attention weights.

        Parameters
        ----------
        attended : :class:`~tensor.TensorVariable`
            The attended sequence, time is the 1-st dimension.

        """
        return self.attention.preprocess(attended)

    @application(outputs=do_apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        initial_state = attended[0,0, :, -self.state_dim:];
        return initial_state

class Decoder(Initializable):
    """Decoder of RNNsearch model."""

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, match_function='SumMacthFunction',
                 use_doubly_stochastic=False, lambda_ds=0.001,
                 use_local_attention=False, window_size=10,
                 use_step_decay_cost=False,
                 use_concentration_cost=False, lambda_ct=10,
                 use_stablilizer=False, lambda_st=50,
                 theano_seed=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.theano_seed = theano_seed

        # Initialize gru with special initial state
        self.transition = GRU(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        self.context_transition= SentenceEncoder(embedding_dim=2*self.state_dim, state_dim=self.state_dim, name='context_transition')

        self.energy_computer = globals()[match_function](name='energy_comp')

        # Initialize the attention mechanism
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=state_dim,
            match_dim=state_dim,
            energy_computer=self.energy_computer,
            use_local_attention=use_local_attention,
            window_size=window_size,
            name="attention")

        # Initialize the readout, note that SoftmaxEmitter emits -1 for
        # initial outputs which is used by LookupFeedBackWMT15
        readout = Readout(
            source_names=['states', 'feedback',
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1, theano_seed=theano_seed),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Linear(input_dim=embedding_dim, name='softmax1').apply]),
            merged_dim=state_dim)

        # Build sequence generator accordingly
        self.sequence_generator = SequenceGenerator(
            representation_dim=self.representation_dim,
            readout=readout,
            transition=self.transition,
            context_transition=self.context_transition,
            attention=self.attention,
            use_step_decay_cost=use_step_decay_cost,
            use_doubly_stochastic=use_doubly_stochastic, lambda_ds=lambda_ds,
            use_concentration_cost=use_concentration_cost, lambda_ct=lambda_ct,
            use_stablilizer=use_stablilizer, lambda_st=lambda_st,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear())
        )

        self.children = [self.sequence_generator]

    # @application(inputs=['representation', 'source_sentence_mask',
    #                      'target_sentence', 'target_sentence_mask'],
    #              outputs=['costs','weights'])
    # def cost(self, representation, source_sentence_mask,
    #          target_sentence, target_sentence_mask):
    #
    #     source_sentence_mask=source_sentence_mask.T
    #     target_sentence = target_sentence.T
    #     target_sentence_mask = target_sentence_mask.T
    #
    #     # Get the cost matrix
    #     cost = self.sequence_generator.cost_matrix(**{
    #         'mask': target_sentence_mask,
    #         'outputs': target_sentence,
    #         'attended': representation,
    #         'attended_mask': source_sentence_mask}
    #     )
    #
    #     '''
    #     return (cost * target_sentence_mask).sum() / \
    #         target_sentence_mask.shape[1]
    #     '''
    #     return (cost * target_sentence_mask).sum()/tensor.sum(target_sentence_mask);
    #     #return cost.sum()/tensor.sum(target_sentence_mask),tensor.sum(target_sentence_mask)

    @application(inputs=['representation_list', 'sentence_mask_list',
                         'target_sentence', 'target_sentence_mask'],
                 outputs=['costs','weights'])
    def cost(self, representation_list, sentence_mask_list,
             target_sentence, target_sentence_mask):

        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(**{
            'mask': target_sentence_mask,
            'outputs': target_sentence,
            'attended_list': representation_list,
            'attended_mask_list': sentence_mask_list}
        )

        '''
        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1]
        '''

        #return (cost * target_sentence_mask).sum()/target_sentence_mask.sum();
        return (cost * target_sentence_mask).sum();
        #return cost.sum()/tensor.sum(target_sentence_mask),tensor.sum(target_sentence_mask)

    @application
    def generate(self, representation, sentence_masks_list,**kwargs):
        return self.sequence_generator.generate(
            n_steps=2 * representation.shape[1],
            batch_size=representation.shape[2],
            attended_list=representation,
            attended_mask_list=sentence_masks_list,
            **kwargs)