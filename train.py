from __future__ import print_function

import logging
import time
import numpy
import os
import cPickle
import string

from collections import Counter
from theano import tensor, function
from toolz import merge
from progressbar import ProgressBar

from blocks.algorithms import (GradientDescent, StepClipping,
                               AdaDelta, AdaGrad, Scale, CompositeRule)
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from search import BeamSearch
from blocks.select import Selector

from checkpoint import CheckpointNMT, LoadNMT
from model_with_l import BidirectionalEncoder, SentenceEncoder,Decoder
from sampling import BleuValidator, Sampler, SamplingBase,pplValidation
from stream import (get_tr_stream, get_tr_stream_withContext,get_dev_stream_withContext, get_dev_stream_with_grdTruth,
    get_test_stream_withContext_grdTruth,get_dev_stream_withContext_grdTruth,
    get_tr_stream_unsorted, _ensure_special_tokens)
from SimplePrinting import SimplePrinting
from learning_rate_halver import (LearningRateHalver, 
                                  LearningRateDoubler, 
                                  OldModelRemover)
from afterprocess import afterprocesser

try:
    from blocks.extras.extensions.plot import Plot
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

logger = logging.getLogger(__name__)


def main(mode, config, use_bokeh=False):

    # Construct model
    logger.info('Building RNN encoder-decoder')
    encoder = BidirectionalEncoder(
        config['src_vocab_size'], config['enc_embed'], config['enc_nhids'],name='word_encoder')
    decoder = Decoder(vocab_size=config['trg_vocab_size'],
                      embedding_dim=config['dec_embed'],
                      state_dim=config['dec_nhids'],
                      representation_dim=config['enc_nhids'] * 2,
                      match_function=config['match_function'],
                      use_doubly_stochastic=config['use_doubly_stochastic'],
                      lambda_ds=config['lambda_ds'],
                      use_local_attention=config['use_local_attention'],
                      window_size=config['window_size'],
                      use_step_decay_cost=config['use_step_decay_cost'],
                      use_concentration_cost=config['use_concentration_cost'],
                      lambda_ct=config['lambda_ct'],
                      use_stablilizer=config['use_stablilizer'],
                      lambda_st=config['lambda_st'])
    # here attended dim (representation_dim) of decoder is 2*enc_nhinds
    # because the context given by the encoder is a bidirectional context

    if mode == "train":

        # Create Theano variables
        logger.info('Creating theano variables')
        context_sentences=[];
        context_sentence_masks=[];
        for i in range(config['ctx_num']):
            context_sentences.append(tensor.lmatrix('context_'+str(i)));
            context_sentence_masks.append(tensor.matrix('context_'+str(i)+'_mask'));
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')
        target_sentence = tensor.lmatrix('target')
        target_sentence_mask = tensor.matrix('target_mask')
        sampling_input = tensor.lmatrix('input')
        dev_source = tensor.lmatrix('dev_source')
        dev_target=tensor.lmatrix('dev_target')

        # Get training and development set streams
        tr_stream = get_tr_stream_withContext(**config)
        dev_stream = get_dev_stream_with_grdTruth(**config)

        # Get cost of the model
        sentence_representations_list=encoder.apply(source_sentence, source_sentence_mask);
        sentence_representations_list=sentence_representations_list.dimshuffle(['x',0,1,2]);
        sentence_masks_list=source_sentence_mask.T.dimshuffle(['x',0,1]);
        for i in range(config['ctx_num']):
            tmp_rep=encoder.apply(context_sentences[i],context_sentence_masks[i]);
            tmp_rep=tmp_rep.dimshuffle(['x',0,1,2]);
            sentence_representations_list=tensor.concatenate([sentence_representations_list,tmp_rep],axis=0);
            sentence_masks_list=tensor.concatenate([sentence_masks_list,context_sentence_masks[i].T.dimshuffle(['x',0,1])],axis=0);


        cost = decoder.cost(sentence_representations_list,
                            sentence_masks_list,
                            target_sentence,
                            target_sentence_mask)

        logger.info('Creating computational graph')
        perplexity = tensor.exp(cost)
        perplexity.name = 'perplexity'
        costs_computer = function(context_sentences+context_sentence_masks+[target_sentence,
                                   target_sentence_mask,
                                   source_sentence,
                                   source_sentence_mask], (perplexity))
        cg = ComputationGraph(cost)

        # Initialize model
        logger.info('Initializing model')
        encoder.weights_init =decoder.weights_init = IsotropicGaussian(
            config['weight_scale'])
        encoder.biases_init =decoder.biases_init = Constant(0)
        encoder.push_initialization_config()
        decoder.push_initialization_config()
        encoder.bidir.prototype.weights_init = Orthogonal()
        decoder.transition.weights_init = Orthogonal()
        encoder.initialize()
        decoder.initialize()

        # apply dropout for regularization
        if config['dropout'] < 1.0:
            # dropout is applied to the output of maxout in ghog
            logger.info('Applying dropout')
            dropout_inputs = [x for x in cg.intermediary_variables
                              if x.name == 'maxout_apply_output']
            cg = apply_dropout(cg, dropout_inputs, config['dropout'])

        # Apply weight noise for regularization
        if config['weight_noise_ff'] > 0.0:
            logger.info('Applying weight noise to ff layers')
            enc_params = Selector(encoder.lookup).get_params().values()
            enc_params += Selector(encoder.fwd_fork).get_params().values()
            enc_params += Selector(encoder.back_fork).get_params().values()
            dec_params = Selector(
                decoder.sequence_generator.readout).get_params().values()
            dec_params += Selector(
                decoder.sequence_generator.fork).get_params().values()
            dec_params += Selector(decoder.state_init).get_params().values()
            cg = apply_noise(
                cg, enc_params+dec_params, config['weight_noise_ff'])


        # Print shapes
        shapes = [param.get_value().shape for param in cg.parameters]
        logger.info("Parameter shapes: ")
        for shape, count in Counter(shapes).most_common():
            logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

        # Print parameter names
        enc_dec_param_dict = merge(Selector(encoder).get_parameters(),
                                   Selector(decoder).get_parameters())
        logger.info("Parameter names: ")
        for name, value in enc_dec_param_dict.items():
            logger.info('    {:15}: {}'.format(value.get_value().shape, name))
        logger.info("Total number of parameters: {}"
                    .format(len(enc_dec_param_dict)))


        # Set up training model
        logger.info("Building model")
        training_model = Model(cost)

        # Set extensions
        logger.info("Initializing extensions")
        extensions = [
            FinishAfter(after_n_batches=config['finish_after']),
            TrainingDataMonitoring([perplexity], after_batch=True),
            CheckpointNMT(config['saveto'],
                          config['model_name'],
                          every_n_batches=config['save_freq'])
        ]

        # Set up beam search and sampling computation graphs if necessary
        if config['hook_samples'] >= 1 or config['bleu_script'] is not None:
            logger.info("Building sampling model")
            sampling_representation = encoder.apply(
                sampling_input, tensor.ones(sampling_input.shape))
            generated = decoder.generate(
                sampling_input, sampling_representation)
            search_model = Model(generated)
            _, samples = VariableFilter(
                bricks=[decoder.sequence_generator], name="outputs")(
                    ComputationGraph(generated[1]))

        # Add sampling
        if config['hook_samples'] >= 1:
            logger.info("Building sampler")
            extensions.append(
                Sampler(model=search_model, data_stream=tr_stream,
                        model_name=config['model_name'],
                        hook_samples=config['hook_samples'],
                        every_n_batches=config['sampling_freq'],
                        src_vocab_size=config['src_vocab_size']))

        # Add early stopping based on bleu
        if False:
            logger.info("Building bleu validator")
            extensions.append(
                BleuValidator(sampling_input, samples=samples, config=config,
                              model=search_model, data_stream=dev_stream,
                              normalize=config['normalized_bleu'],
                              every_n_batches=config['bleu_val_freq'],
                              n_best=3,
                              track_n_models=6))

        logger.info("Building perplexity validator")
        extensions.append(
                pplValidation(dev_source,dev_target, config=config,
                        model=costs_computer, data_stream=dev_stream,
                        model_name=config['model_name'],
                        every_n_batches=config['sampling_freq']))


        # Plot cost in bokeh if necessary
        if use_bokeh and BOKEH_AVAILABLE:
            extensions.append(
                Plot('Cs-En', channels=[['decoder_cost_cost']],
                     after_batch=True))

        # Reload model if necessary
        if config['reload']:
            extensions.append(LoadNMT(config['saveto']))

        initial_learning_rate = config['initial_learning_rate']
        log_path = os.path.join(config['saveto'], 'log')
        if config['reload'] and os.path.exists(log_path):
            with open(log_path, 'rb') as source:
                log = cPickle.load(source)
                last = max(log.keys()) - 1
                if 'learning_rate' in log[last]:
                    initial_learning_rate = log[last]['learning_rate']

        # Set up training algorithm
        logger.info("Initializing training algorithm")
        algorithm = GradientDescent(
            cost=cost, parameters=cg.parameters,
            step_rule=CompositeRule([Scale(initial_learning_rate),
                                     StepClipping(config['step_clipping']),
                                     eval(config['step_rule'])()]))

        _learning_rate = algorithm.step_rule.components[0].learning_rate
        if config['learning_rate_decay']:
            extensions.append(
                LearningRateHalver(record_name='validation_cost',
                                   comparator=lambda x, y: x > y,
                                   learning_rate=_learning_rate,
                                   patience_default=3))
        else:
            extensions.append(OldModelRemover(saveto=config['saveto']))

        if config['learning_rate_grow']:
            extensions.append(
                LearningRateDoubler(record_name='validation_cost',
                                    comparator=lambda x, y: x < y,
                                    learning_rate=_learning_rate,
                                    patience_default=3))

        extensions.append(
            SimplePrinting(config['model_name'], after_batch=True))

        # Initialize main loop
        logger.info("Initializing main loop")
        main_loop = MainLoop(
            model=training_model,
            algorithm=algorithm,
            data_stream=tr_stream,
            extensions=extensions
        )

        # Train!
        main_loop.run()

    elif mode == 'ppl':
        # Create Theano variables
        # Create Theano variables
        logger.info('Creating theano variables')
        context_sentences=[];
        context_sentence_masks=[];
        for i in range(config['ctx_num']):
            context_sentences.append(tensor.lmatrix('context_'+str(i)));
            context_sentence_masks.append(tensor.matrix('context_'+str(i)+'_mask'));
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')
        target_sentence = tensor.lmatrix('target')
        target_sentence_mask = tensor.matrix('target_mask')

        # Get training and development set streams
        #tr_stream = get_tr_stream_withContext(**config)
        dev_stream = get_dev_stream_withContext_grdTruth(**config)

        # Get cost of the model
        sentence_representations_list=encoder.apply(source_sentence, source_sentence_mask);
        sentence_representations_list=sentence_representations_list.dimshuffle(['x',0,1,2]);
        sentence_masks_list=source_sentence_mask.T.dimshuffle(['x',0,1]);
        for i in range(config['ctx_num']):
            tmp_rep=encoder.apply(context_sentences[i],context_sentence_masks[i]);
            tmp_rep=tmp_rep.dimshuffle(['x',0,1,2]);
            sentence_representations_list=tensor.concatenate([sentence_representations_list,tmp_rep],axis=0);
            sentence_masks_list=tensor.concatenate([sentence_masks_list,context_sentence_masks[i].T.dimshuffle(['x',0,1])],axis=0);


        cost = decoder.cost(sentence_representations_list,
                            sentence_masks_list,
                            target_sentence,
                            target_sentence_mask)

        logger.info('Creating computational graph')
        costs_computer = function(context_sentences+context_sentence_masks+[target_sentence,
                                   target_sentence_mask,
                                   source_sentence,
                                   source_sentence_mask], (cost))


        logger.info("Loading the model..")
        model = Model(cost)
        #loader = LoadNMT(config['saveto'])
        loader = LoadNMT(config['validation_load']);
        loader.set_model_parameters(model, loader.load_parameters_default())
        logger.info("Started Validation: ")

        ts = dev_stream.get_epoch_iterator()
        total_cost = 0.0
        total_tokens=0.0
        #pbar = ProgressBar(max_value=len(ts)).start()#modified
        pbar = ProgressBar(max_value=10000).start();
        for i, (ctx_0,ctx_0_mask,ctx_1,ctx_1_mask,ctx_2,ctx_2_mask,src, src_mask, trg, trg_mask) in enumerate(ts):
            costs  = costs_computer(*[ctx_0,ctx_1,ctx_2,ctx_0_mask,ctx_1_mask,ctx_2_mask,trg, trg_mask,src, src_mask])
            cost = costs.sum()
            total_cost+=cost
            total_tokens+=trg_mask.sum()
            pbar.update(i + 1)
        total_cost/=total_tokens;
        pbar.finish()
        #dev_stream.reset()

        # run afterprocess
        # self.ap.main()
        total_cost=2**total_cost;
        print("Average validation cost: " + str(total_cost));
    elif mode == 'translate':

        logger.info('Creating theano variables')
        context_sentences=[];
        context_sentence_masks=[];
        for i in range(config['ctx_num']):
            context_sentences.append(tensor.lmatrix('context_'+str(i)));
            context_sentence_masks.append(tensor.matrix('context_'+str(i)+'_mask'));
        source_sentence = tensor.lmatrix('source')
        source_sentence_mask = tensor.matrix('source_mask')

        sutils = SamplingBase()
        unk_idx = config['unk_id']
        src_eos_idx = config['src_vocab_size'] - 1
        trg_eos_idx = config['trg_vocab_size'] - 1
        trg_vocab = _ensure_special_tokens(
            cPickle.load(open(config['trg_vocab'], 'rb')), bos_idx=0,
            eos_idx=trg_eos_idx, unk_idx=unk_idx)
        trg_ivocab = {v: k for k, v in trg_vocab.items()}
        config['batch_size'] = 1

        sentence_representations_list=encoder.apply(source_sentence, source_sentence_mask);
        sentence_representations_list=sentence_representations_list.dimshuffle(['x',0,1,2]);
        sentence_masks_list=source_sentence_mask.T.dimshuffle(['x',0,1]);
        for i in range(config['ctx_num']):
            tmp_rep=encoder.apply(context_sentences[i],context_sentence_masks[i]);
            tmp_rep=tmp_rep.dimshuffle(['x',0,1,2]);
            sentence_representations_list=tensor.concatenate([sentence_representations_list,tmp_rep],axis=0);
            sentence_masks_list=tensor.concatenate([sentence_masks_list,context_sentence_masks[i].T.dimshuffle(['x',0,1])],axis=0);
        generated = decoder.generate(sentence_representations_list,sentence_masks_list)
        _, samples = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))  # generated[1] is next_outputs
        beam_search = BeamSearch(samples=samples)

        logger.info("Loading the model..")
        model = Model(generated)
        #loader = LoadNMT(config['saveto'])
        loader = LoadNMT(config['validation_load']);
        loader.set_model_parameters(model, loader.load_parameters_default())

        logger.info("Started translation: ")
        test_stream = get_dev_stream_withContext(**config)
        ts = test_stream.get_epoch_iterator()
        rts = open(config['val_set_source']).readlines()
        ftrans_original = open(config['val_output_orig'], 'w')
        saved_weights = []
        total_cost = 0.0

        pbar = ProgressBar(max_value=len(rts)).start()
        for i, (line, line_raw) in enumerate(zip(ts, rts)):
            trans_in = line_raw[3].split()
            seqs=[];
            input_=[];
            input_mask=[];
            for j in range(config['ctx_num']+1):
                seqs.append(sutils._oov_to_unk(
                    line[2*j][0], config['src_vocab_size'], unk_idx))
                input_mask.append(numpy.tile(line[2*j+1][0],(config['beam_size'], 1)))
                input_.append(numpy.tile(seqs[j], (config['beam_size'], 1)))
            #v=costs_computer(input_[0]);
            # draw sample, checking to ensure we don't get an empty string back
            trans, costs, attendeds, weights = \
                beam_search.search(
                    input_values={source_sentence: input_[3],source_sentence_mask:input_mask[3],
                                  context_sentences[0]: input_[0],context_sentence_masks[0]:input_mask[0],
                                  context_sentences[1]: input_[1],context_sentence_masks[1]:input_mask[1],
                                  context_sentences[2]: input_[2],context_sentence_masks[2]:input_mask[2]},
                    max_length=3*len(seqs[2]), eol_symbol=trg_eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if config['normalized_bleu']:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            b = numpy.argsort(costs)[0]
            #best=numpy.argsort(costs)[0:config['beam_size']];
            #for b in best:
            try:
                total_cost += costs[b]
                trans_out = trans[b]
                totalLen=4*len(line[0][0]);
                #weight = weights[b][:, :totalLen]
                weight=weights
                trans_out = sutils._idx_to_word(trans_out, trg_ivocab)
            except ValueError:
                logger.info(
                    "Can NOT find a translation for line: {}".format(i+1))
                trans_out = '<UNK>'
            saved_weights.append(weight)
            print(' '.join(trans_out), file=ftrans_original)
            pbar.update(i + 1)

        pbar.finish()
        logger.info("Total cost of the test: {}".format(total_cost))
        cPickle.dump(saved_weights, open(config['attention_weights'], 'wb'))
        ftrans_original.close()
        ap = afterprocesser(config)
        ap.main()


