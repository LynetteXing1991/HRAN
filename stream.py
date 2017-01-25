import numpy

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from six.moves import cPickle

import logging
logger = logging.getLogger(__name__)
def _ensure_special_tokens(vocab, bos_idx=0, eos_idx=0, unk_idx=1):
    """Ensures special tokens exist in the dictionary."""

    # remove tokens if they exist in some other index
    tokens_to_remove = [k for k, v in vocab.items()
                        if v in [bos_idx, eos_idx, unk_idx]]
    for token in tokens_to_remove:
        vocab.pop(token)
    # put corresponding item
    vocab['<S>'] = bos_idx
    vocab['</S>'] = eos_idx
    vocab['<UNK>'] = unk_idx
    return vocab


def _length(sentence_pair):
    """Assumes target is the last element in the tuple."""
    return len(sentence_pair[-1])


class PaddingWithEOS(Padding):
    """Padds a stream with given end of sequence idx."""
    def __init__(self, data_stream, eos_idx, **kwargs):
        kwargs['data_stream'] = data_stream
        self.eos_idx = eos_idx
        super(PaddingWithEOS, self).__init__(**kwargs)

    def get_data_from_batch(self, request=None):
        if request is not None:
            raise ValueError
        data = list(next(self.child_epoch_iterator))
        data_with_masks = []
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, data)):
            if source not in self.mask_sources:
                data_with_masks.append(source_data)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_data[0]).dtype

            padded_data = numpy.ones(
                (len(source_data), max_sequence_length) + rest_shape,
                dtype=dtype) * self.eos_idx[i]
            for i, sample in enumerate(source_data):
                padded_data[i, :len(sample)] = sample
            data_with_masks.append(padded_data)

            mask = numpy.zeros((len(source_data), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            data_with_masks.append(mask)
        return tuple(data_with_masks)

class PaddingWithEOSContext(Padding):
    """Padds a stream with given end of sequence idx."""
    def __init__(self, data_stream, eos_idx, **kwargs):
        kwargs['data_stream'] = data_stream
        self.eos_idx = eos_idx
        super(PaddingWithEOSContext, self).__init__(**kwargs)

    def get_data_from_batch(self, request=None):
        if request is not None:
            raise ValueError
        data = list(next(self.child_epoch_iterator))
        data_with_masks = []
        max_sequence_length=0;
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, data)):
            # if source not in self.mask_sources:
            #     data_with_masks.append(source_data)
            #     continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            tmp_ml=max(lengths);
            max_sequence_length = max([max_sequence_length,tmp_ml])
        logger.info("max sequence length: "+str(max_sequence_length));
        print("max sequence length: "+str(max_sequence_length));
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, data)):
            if source not in self.mask_sources:
                data_with_masks.append(source_data)
                continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            #max_sequence_length = max([max_sequence_length,max(lengths)])
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_data[0]).dtype

            padded_data = numpy.ones(
                (len(source_data), max_sequence_length) + rest_shape,
                dtype=dtype) * self.eos_idx[i]
            for i, sample in enumerate(source_data):
                padded_data[i, :len(sample)] = sample
            data_with_masks.append(padded_data)

            mask = numpy.zeros((len(source_data), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            data_with_masks.append(mask)
        return tuple(data_with_masks)

    def transform_batch(self, batch):
        batch_with_masks = []
        max_sequence_length=0;
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, batch)):
            # if source not in self.mask_sources:
            #     data_with_masks.append(source_data)
            #     continue

            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]
            tmp_ml=max(lengths);
            max_sequence_length = max([max_sequence_length,tmp_ml])

        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue
            shapes = [numpy.asarray(sample).shape for sample in source_batch]
            lengths = [shape[0] for shape in shapes]
            #max_sequence_length = max(lengths)
            rest_shape = shapes[0][1:]
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_batch[0]).dtype

            padded_batch = numpy.zeros(
                (len(source_batch), max_sequence_length) + rest_shape,
                dtype=dtype)
            for i, sample in enumerate(source_batch):
                padded_batch[i, :len(sample)] = sample
            batch_with_masks.append(padded_batch)

            mask = numpy.zeros((len(source_batch), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)


class _oov_to_unk(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, ctx_num,src_vocab_size=30000, trg_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id
        self.ctx_num=ctx_num

    def __call__(self, sentence_pair):
        return tuple([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[i]] for i in range(self.ctx_num))\
               +([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[self.ctx_num]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[self.ctx_num+1]])

class _oov_to_unk_posTag(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, ctx_num,src_vocab_size=30000, trg_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id
        self.ctx_num=ctx_num

    def __call__(self, sentence_pair):
        return tuple([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[i]] for i in range(2*self.ctx_num+1))\
               +([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[2*self.ctx_num+1]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[2*self.ctx_num+2]])

class _oov_to_unk_posTag_dev(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, ctx_num,src_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.unk_id = unk_id
        self.ctx_num=ctx_num

    def __call__(self, sentence_pair):
        return tuple([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[i]] for i in range(2*self.ctx_num+1))\
               +([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[2*self.ctx_num+1]],)

class _oov_to_unk_dev(object):
    """Maps out of vocabulary token index to unk token index."""
    def __init__(self, ctx_num,src_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.unk_id = unk_id
        self.ctx_num=ctx_num

    def __call__(self, sentence_pair):
        return tuple([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[i]] for i in range(self.ctx_num))\
               +([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[self.ctx_num]],)

class _too_long(object):
    """Filters sequences longer than given sequence length."""
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentence_pair])

def get_tr_stream_withContext(src_vocab, trg_vocab, ctx_datas,ctx_num,src_data, trg_data,
                  src_vocab_size=30000, trg_vocab_size=30000, unk_id=1,
                  seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""

    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab, 'rb')),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = _ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab, 'rb')),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # Get text files from both source and target
    ctx_datasets=[];
    for i in range(ctx_num):
        ctx_datasets.append(TextFile([ctx_datas[i]], src_vocab, None))
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)

    # Merge them to get a source, target pair
    stream = Merge([i.get_example_stream() for i in ctx_datasets]+[src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   tuple('context_'+str(i) for i in range(ctx_num))+('source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))

    #Replace out of vocabulary tokens with unk token
    stream = Mapping(stream,
                     _oov_to_unk(ctx_num=ctx_num,
                                 src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       batch_size*sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(
        stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    masked_stream = PaddingWithEOSContext(
        stream, [src_vocab_size - 1 for i in range(ctx_num+1)]+[trg_vocab_size - 1])

    return masked_stream

def get_tr_stream_withContext_withPosTag(src_vocab, trg_vocab, ctx_datas,posTag_datas,ctx_num,src_data, trg_data,
                  src_vocab_size=30000, trg_vocab_size=30000, unk_id=1,
                  seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""

    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab, 'rb')),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = _ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab, 'rb')),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # Get text files from both source and target
    ctx_datasets=[];
    posTag_datasets=[];
    for i in range(ctx_num):
        ctx_datasets.append(TextFile([ctx_datas[i]], src_vocab, None))
        posTag_datasets.append(TextFile([posTag_datas[i]],src_vocab,None))
    posTag_datasets.append(TextFile([posTag_datas[ctx_num]],src_vocab,None))
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)

    # Merge them to get a source, target pair
    stream = Merge([i.get_example_stream() for i in ctx_datasets]+
                   [i.get_example_stream() for i in posTag_datasets]+
                   [src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   tuple('context_'+str(i) for i in range(ctx_num))+
                   tuple('context_posTag_'+str(i) for i in range(ctx_num))
                   +('source_posTag','source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))

    #Replace out of vocabulary tokens with unk token
    stream = Mapping(stream,
                     _oov_to_unk_posTag(ctx_num=ctx_num,
                                 src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       batch_size*sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(
        stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    masked_stream = PaddingWithEOSContext(
        stream, [src_vocab_size - 1 for i in range(2*ctx_num+2)]+[trg_vocab_size - 1])

    return masked_stream


def get_tr_stream(src_vocab, trg_vocab, src_data, trg_data,
                  src_vocab_size=30000, trg_vocab_size=30000, unk_id=1,
                  seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""

    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab, 'rb')),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = _ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab, 'rb')),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)

    # Merge them to get a source, target pair
    stream = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))

    # Replace out of vocabulary tokens with unk token
    stream = Mapping(stream,
                     _oov_to_unk(src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       batch_size*sort_k_batches))

    # Sort all samples in the read-ahead batch
    stream = Mapping(stream, SortMapping(_length))

    # Convert it into a stream again
    stream = Unpack(stream)

    # Construct batches from the stream with specified batch size
    stream = Batch(
        stream, iteration_scheme=ConstantScheme(batch_size))

    # Pad sequences that are short
    masked_stream = PaddingWithEOS(
        stream, [src_vocab_size - 1, trg_vocab_size - 1])

    return masked_stream


def get_dev_stream(val_set_source=None, src_vocab=None, src_vocab_size=30000,
                   unk_id=1, **kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if val_set_source is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        print val_set_source, type(src_vocab)
        dev_dataset = TextFile([val_set_source], src_vocab, None)
        dev_stream = DataStream(dev_dataset)
    return dev_stream

def get_dev_stream_withContext_grdTruth(val_ctx_datas=None,val_set_source=None,val_set_target=None,src_vocab=None, src_vocab_size=30000,trg_vocab=None,trg_vocab_size=30000,batch_size=128,
                   unk_id=1,ctx_num=3, **kwargs):
    """Setup development set stream if necessary."""
    masked_stream = None
    if val_set_source is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        trg_vocab = _ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            cPickle.load(open(trg_vocab, 'rb')),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)
        print val_set_source, type(src_vocab)
        # Get text files from both source and target
        ctx_datasets=[];
        for i in range(ctx_num):
            ctx_datasets.append(TextFile([val_ctx_datas[i]], src_vocab, None))
        dev_dataset = TextFile([val_set_source], src_vocab, None)
        dev_target = TextFile([val_set_target], trg_vocab, None)
        dev_stream = Merge([i.get_example_stream() for i in ctx_datasets]+[dev_dataset.get_example_stream(),dev_target.get_example_stream()],
                   tuple('context_'+str(i) for i in range(ctx_num))+('source','target'))
        stream = Mapping(dev_stream,
                         _oov_to_unk(ctx_num=ctx_num,
                                 src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

        # Build a batched version of stream to read k batches ahead
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(batch_size))
        masked_stream = PaddingWithEOSContext(
            stream, [src_vocab_size - 1 for i in range(ctx_num+1)]+[trg_vocab_size - 1])

    return masked_stream

def get_test_stream_withContext_grdTruth(test_ctx_datas=None,test_set_source=None,test_set_target=None,src_vocab=None, src_vocab_size=30000,trg_vocab=None,trg_vocab_size=30000,batch_size=128,
                   unk_id=1,ctx_num=3, **kwargs):
    """Setup development set stream if necessary."""
    masked_stream = None
    if test_set_source is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        trg_vocab = _ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            cPickle.load(open(trg_vocab, 'rb')),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)
        print test_set_source, type(src_vocab)
        # Get text files from both source and target
        ctx_datasets=[];
        for i in range(ctx_num):
            ctx_datasets.append(TextFile([test_ctx_datas[i]], src_vocab, None))
        dev_dataset = TextFile([test_set_source], src_vocab, None)
        dev_target = TextFile([test_set_target], trg_vocab, None)
        dev_stream = Merge([i.get_example_stream() for i in ctx_datasets]+[dev_dataset.get_example_stream(),dev_target.get_example_stream()],
                   tuple('context_'+str(i) for i in range(ctx_num))+('source','target'))
        stream = Mapping(dev_stream,
                         _oov_to_unk(ctx_num=ctx_num,
                                 src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

        # Build a batched version of stream to read k batches ahead
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(batch_size))
        masked_stream = PaddingWithEOSContext(
            stream, [src_vocab_size - 1 for i in range(ctx_num+1)]+[trg_vocab_size - 1])

    return masked_stream

def get_dev_stream_withContext_withPosTag(test_ctx_datas=None,test_posTag_datas=None,test_set_source=None, src_vocab=None, src_vocab_size=30000,
                   unk_id=1,ctx_num=3, **kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if test_set_source is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        print test_set_source, type(src_vocab)
        # Get text files from both source and target
        ctx_datasets=[];
        posTag_datasets=[];
        for i in range(ctx_num):
            ctx_datasets.append(TextFile([test_ctx_datas[i]], src_vocab, None))
            posTag_datasets.append(TextFile([test_posTag_datas[i]],src_vocab,None))
        posTag_datasets.append(TextFile([test_posTag_datas[ctx_num]],src_vocab,None))
        src_dataset = TextFile([test_set_source], src_vocab, None)

        # Merge them to get a source, target pair
        dev_stream = Merge([i.get_example_stream() for i in ctx_datasets]+
                       [i.get_example_stream() for i in posTag_datasets]+
                       [src_dataset.get_example_stream()],
                       tuple('context_'+str(i) for i in range(ctx_num))+
                       tuple('context_posTag_'+str(i) for i in range(ctx_num))
                       +('source_posTag','source'))

        stream = Mapping(dev_stream,
                         _oov_to_unk_posTag_dev(ctx_num=ctx_num,
                                 src_vocab_size=src_vocab_size,
                                 unk_id=unk_id))

        # Build a batched version of stream to read k batches ahead
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(1))
        masked_stream = PaddingWithEOSContext(
            stream, [src_vocab_size - 1 for i in range(2*ctx_num+2)])

    return masked_stream

def get_dev_stream_withContext_withPosTag_grd(dev_ctx_datas=None,dev_posTag_datas=None,dev_set_source=None,dev_set_target=None,
                                              src_vocab=None, src_vocab_size=30000,trg_vocab=None, trg_vocab_size=30000,
                   unk_id=1,ctx_num=3, batch_size=80,**kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if dev_set_source is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        trg_vocab = _ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            cPickle.load(open(trg_vocab, 'rb')),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)
        print dev_set_source, type(src_vocab)
        # Get text files from both source and target
        ctx_datasets=[];
        posTag_datasets=[];
        for i in range(ctx_num):
            ctx_datasets.append(TextFile([dev_ctx_datas[i]], src_vocab, None))
            posTag_datasets.append(TextFile([dev_posTag_datas[i]],src_vocab,None))
        posTag_datasets.append(TextFile([dev_posTag_datas[ctx_num]],src_vocab,None))
        src_dataset = TextFile([dev_set_source], src_vocab, None)
        trg_dataset = TextFile([dev_set_target], trg_vocab, None)
        # Merge them to get a source, target pair
        dev_stream = Merge([i.get_example_stream() for i in ctx_datasets]+
                       [i.get_example_stream() for i in posTag_datasets]+
                       [src_dataset.get_example_stream(),trg_dataset.get_example_stream()],
                       tuple('context_'+str(i) for i in range(ctx_num))+
                       tuple('context_posTag_'+str(i) for i in range(ctx_num))
                       +('source_posTag','source','target'))

        stream = Mapping(dev_stream,
                         _oov_to_unk_posTag(ctx_num=ctx_num,
                                 src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

        # Build a batched version of stream to read k batches ahead
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(batch_size=batch_size))
        masked_stream = PaddingWithEOSContext(
            stream, [src_vocab_size - 1 for i in range(2*ctx_num+2),trg_vocab_size-1])

    return masked_stream

def get_test_stream_withContext_withPosTag_grd(test_ctx_datas=None,test_posTag_datas=None,test_set_source=None,test_set_target=None,
                                              src_vocab=None, src_vocab_size=30000,trg_vocab=None, trg_vocab_size=30000,
                   unk_id=1,ctx_num=3, batch_size=80,**kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if test_set_source is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        trg_vocab = _ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            cPickle.load(open(trg_vocab, 'rb')),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)
        print test_set_source, type(src_vocab)
        # Get text files from both source and target
        ctx_datasets=[];
        posTag_datasets=[];
        for i in range(ctx_num):
            ctx_datasets.append(TextFile([test_ctx_datas[i]], src_vocab, None))
            posTag_datasets.append(TextFile([test_posTag_datas[i]],src_vocab,None))
        posTag_datasets.append(TextFile([test_posTag_datas[ctx_num]],src_vocab,None))
        src_dataset = TextFile([test_set_source], src_vocab, None)
        trg_dataset = TextFile([test_set_target], trg_vocab, None)
        # Merge them to get a source, target pair
        dev_stream = Merge([i.get_example_stream() for i in ctx_datasets]+
                       [i.get_example_stream() for i in posTag_datasets]+
                       [src_dataset.get_example_stream(),trg_dataset.get_example_stream()],
                       tuple('context_'+str(i) for i in range(ctx_num))+
                       tuple('context_posTag_'+str(i) for i in range(ctx_num))
                       +('source_posTag','source','target'))

        stream = Mapping(dev_stream,
                         _oov_to_unk_posTag(ctx_num=ctx_num,
                                 src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

        # Build a batched version of stream to read k batches ahead
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(batch_size=batch_size))
        masked_stream = PaddingWithEOSContext(
            stream, [src_vocab_size - 1 for i in range(2*ctx_num+2),trg_vocab_size-1])

    return masked_stream



def get_dev_stream_withContext(val_ctx_datas=None,val_set_source=None, src_vocab=None, src_vocab_size=30000,
                   unk_id=1,ctx_num=3, **kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if val_set_source is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        print val_set_source, type(src_vocab)
        # Get text files from both source and target
        ctx_datasets=[];
        for i in range(ctx_num):
            ctx_datasets.append(TextFile([val_ctx_datas[i]], src_vocab, None))
        dev_dataset = TextFile([val_set_source], src_vocab, None)
        dev_stream = Merge([i.get_example_stream() for i in ctx_datasets]+[dev_dataset.get_example_stream()],
                   tuple('context_'+str(i) for i in range(ctx_num))+('source',))
        stream = Mapping(dev_stream,
                         _oov_to_unk_dev(ctx_num=ctx_num,
                                 src_vocab_size=src_vocab_size,
                                 unk_id=unk_id))

        # Build a batched version of stream to read k batches ahead
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(1))
        masked_stream = PaddingWithEOSContext(
        stream, [src_vocab_size - 1 for i in range(ctx_num+1)])

    return masked_stream

def get_dev_stream_with_grdTruth(val_set_source=None, val_set_target=None, src_vocab=None, src_vocab_size=30000,trg_vocab=None,trg_vocab_size=30000,batch_size=128,
                   unk_id=1,seq_len=50, **kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if val_set_source is not None and src_vocab is not None:
        src_vocab = _ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab, 'rb')),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        trg_vocab = _ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            cPickle.load(open(trg_vocab, 'rb')),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

        print val_set_source, type(src_vocab)
        dev_dataset = TextFile([val_set_source], src_vocab, None)
        trg_dataset = TextFile([val_set_target], trg_vocab, None)
        # Merge them to get a source, target pair
        dev_stream = Merge([dev_dataset.get_example_stream(),
                        trg_dataset.get_example_stream()],
                       ('dev_source', 'dev_target'))
        # Filter sequences that are too long
        stream = Filter(dev_stream,
                        predicate=_too_long(seq_len=seq_len))

        # Replace out of vocabulary tokens with unk token
        stream = Mapping(stream,
                         _oov_to_unk(src_vocab_size=src_vocab_size,
                                     trg_vocab_size=trg_vocab_size,
                                     unk_id=unk_id))

        # Build a batched version of stream to read k batches ahead
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(1))
        # Pad sequences that are short
        masked_stream = PaddingWithEOS(
            stream, [src_vocab_size - 1, trg_vocab_size - 1])
    return masked_stream

def get_tr_stream_unsorted(src_vocab, trg_vocab, src_data, trg_data,
                  src_vocab_size=30000, trg_vocab_size=30000, unk_id=1,
                  seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""

    # Load dictionaries and ensure special tokens exist
    src_vocab = _ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab, 'rb')),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = _ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab, 'rb')),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab, None)
    trg_dataset = TextFile([trg_data], trg_vocab, None)

    # Merge them to get a source, target pair
    stream = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream()],
                   ('source', 'target'))

    # Filter sequences that are too long
    stream = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))

    # Replace out of vocabulary tokens with unk token
    stream = Mapping(stream,
                     _oov_to_unk(src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))

    # Build a batched version of stream to read k batches ahead
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(1))

    # Pad sequences that are short
    masked_stream = PaddingWithEOS(
        stream, [src_vocab_size - 1, trg_vocab_size - 1])

    return masked_stream
