from __future__ import print_function

import logging
import numpy
import operator
import os
import re
import signal
import time
import cPickle

from blocks.extensions import SimpleExtension
from search import BeamSearch
from afterprocess import afterprocesser

from subprocess import Popen, PIPE
from progressbar import ProgressBar

logger = logging.getLogger(__name__)


class SamplingBase(object):
    """Utility class for BleuValidator and Sampler."""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq, vocab_size, unk_idx):
        return [x if x < vocab_size else unk_idx for x in seq]

    def _idx_to_sent(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])

    def _idx_to_word(self, seq, ivocab):
        # return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])
        return [ivocab.get(idx, "<UNK>") for idx in seq]


class Sampler(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self, model, data_stream, model_name, hook_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, src_vocab_size=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.hook_samples = hook_samples
        self.data_stream = data_stream
        self.model_name = model_name
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_vocab_size = src_vocab_size
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):

        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
        if not self.src_vocab_size:
            self.src_vocab_size = len(self.src_vocab)

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]
        batch_size = batch['source'].shape[0]
        hook_samples = min(batch_size, self.hook_samples)

        # TODO: this is problematic for boundary conditions, eg. last batch
        sample_idx = numpy.random.choice(
            batch_size, hook_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        print()
        for i in range(hook_samples):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)

            inp = input_[i, :input_length]
            _1, outputs, _2, _3, costs = (self.sampling_fn(inp[None, :]))
            outputs = outputs.flatten()
            costs = costs.T

            sample_length = self._get_true_length(outputs, self.trg_vocab)

            print("Sampling: " + self.model_name)

            print("Input : ", self._idx_to_sent(input_[i][:input_length],
                                                self.src_ivocab))
            print("Target: ", self._idx_to_sent(target_[i][:target_length],
                                                self.trg_ivocab))
            print("Sample: ", self._idx_to_sent(outputs[:sample_length],
                                                self.trg_ivocab))
            print("Sample cost: ", costs[:sample_length].sum())
            print()

class perplexityValidation(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self,source_sentence,samples, model, data_stream, model_name,config,
                 src_vocab=None, n_best=1, track_n_models=1, trg_ivocab=None,
                 patience=10, normalize=True, **kwargs):
        super(perplexityValidation, self).__init__(**kwargs)
        self.model = model
        self.data_stream = data_stream
        self.model_name = model_name
        self.src_vocab = src_vocab
        self.trg_ivocab = trg_ivocab
        self.is_synced = False
        self.sampling_fn = model.get_theano_function()

        self.source_sentence = source_sentence
        self.samples = samples
        self.config = config
        self.n_best = n_best
        self.normalize = normalize
        self.patience = patience

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.trg_ivocab = trg_ivocab
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.src_eos_idx = config['src_vocab_size'] - 1
        self.beam_search = BeamSearch(samples=samples)

    def do(self, which_callback, *args):

        print()
        # Evaluate and save if necessary
        cost = self._evaluate_model()
        print("Average validation cost: " + str(cost));

    def _evaluate_model(self):

        logger.info("Started Validation: ")

        if not self.trg_ivocab:
            sources = self._get_attr_rec(self.main_loop, 'data_stream')
            trg_vocab = sources.data_streams[1].dataset.dictionary
            self.trg_ivocab = {v: k for k, v in trg_vocab.items()}

        ts = self.data_stream.get_epoch_iterator()
        total_cost = 0.0

        #pbar = ProgressBar(max_value=len(ts)).start()#modified
        pbar = ProgressBar(max_value=20036).start();
        for i, line in enumerate(ts):
            seq = self._oov_to_unk(
                line[0], self.config['src_vocab_size'], self.unk_idx)
            input_ = numpy.tile(seq, (self.config['beam_size'], 1))

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs, attendeds, weights = \
                self.beam_search.search(
                    input_values={self.source_sentence: input_},
                    max_length=3*len(seq), eol_symbol=self.src_eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            best = numpy.argsort(costs)[0]
            try:
                total_cost += costs[best]
                trans_out = trans[best]
                trans_out = self._idx_to_word(trans_out, self.trg_ivocab)
            except ValueError:
                logger.info(
                    "Can NOT find a translation for line: {}".format(i+1))
                trans_out = '<UNK>'

            pbar.update(i + 1)

        pbar.finish()
        self.data_stream.reset()

        # run afterprocess
        # self.ap.main()
        self.main_loop.log.current_row['validation_cost'] = total_cost

        return total_cost

class pplValidation(SimpleExtension, SamplingBase):
    """Random Sampling from model."""

    def __init__(self,source_sentence,target_sentence, model, data_stream, model_name,config,
                 src_vocab=None, n_best=1, track_n_models=1, trg_ivocab=None,
                 patience=10, normalize=True, **kwargs):
        super(pplValidation, self).__init__(**kwargs)
        self.model = model
        self.data_stream = data_stream
        self.model_name = model_name
        self.src_vocab = src_vocab
        self.trg_ivocab = trg_ivocab
        self.is_synced = False
        self.sampling_fn = model

        self.source_sentence = source_sentence
        self.target_sentence = target_sentence
        self.config = config
        self.n_best = n_best
        self.normalize = normalize
        self.patience = patience


    def do(self, which_callback, *args):

        print()
        # Evaluate and save if necessary
        cost = self._evaluate_model()
        print("Average validation cost: " + str(cost));

    def _evaluate_model(self):

        logger.info("Started Validation: ")

        ts = self.data_stream.get_epoch_iterator()
        total_cost = 0.0

        #pbar = ProgressBar(max_value=len(ts)).start()#modified
        pbar = ProgressBar(max_value=20050).start();
        for i, (src, src_mask, trg, trg_mask) in enumerate(ts):
            costs  = self.model(*[trg, trg_mask, src, src_mask])
            cost = costs.sum()
            total_cost+=costs
            pbar.update(i + 1)
        total_cost/=20050;
        pbar.finish()
        self.data_stream.reset()

        # run afterprocess
        # self.ap.main()
        self.main_loop.log.current_row['validation_cost'] = total_cost

        return total_cost

class BleuValidator(SimpleExtension, SamplingBase):

    def __init__(self, source_sentence, samples, model, data_stream,
                 config, n_best=1, track_n_models=1, trg_ivocab=None,
                 patience=10, normalize=True, **kwargs):
        super(BleuValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.patience = patience

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.trg_ivocab = trg_ivocab
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.src_eos_idx = config['src_vocab_size'] - 1
        self.best_models = []
        self.beam_search = BeamSearch(samples=samples)
        self.multibleu_cmd = ['perl', self.config['bleu_script'],
                              self.config['val_set_target'], '<']
        self.compbleu_cmd = [self.config['bleu_script_1'], 
                             self.config['val_set_target'],
                             self.config['val_output_repl']]
        self.ap = afterprocesser(config)

        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.config['val_burn_in']:
            return

        # Evaluate and save if necessary
        bleu, cost = self._evaluate_model()
        self._save_model(bleu, cost)
        self._stop()

    def _stop(self):
        def get_last_max(l):
            t = 0
            r = 0
            for i, j in enumerate(l):
                if j >= t:
                    r = i
            return r

    def _evaluate_model(self):

        logger.info("Started Validation: ")

        if not self.trg_ivocab:
            sources = self._get_attr_rec(self.main_loop, 'data_stream')
            trg_vocab = sources.data_streams[1].dataset.dictionary
            self.trg_ivocab = {v: k for k, v in trg_vocab.items()}

        ts = self.data_stream.get_epoch_iterator()
        rts = open(self.config['val_set_source']).readlines()
        ftrans_original = open(self.config['val_output_orig'], 'w')
        saved_weights = []
        total_cost = 0.0

        pbar = ProgressBar(max_value=len(rts)).start()
        for i, (line, line_raw) in enumerate(zip(ts, rts)):
            trans_in = line_raw.split()
            seq = self._oov_to_unk(
                line[0], self.config['src_vocab_size'], self.unk_idx)
            input_ = numpy.tile(seq, (self.config['beam_size'], 1))

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs, attendeds, weights = \
                self.beam_search.search(
                    input_values={self.source_sentence: input_},
                    max_length=3*len(seq), eol_symbol=self.src_eos_idx,
                    ignore_first_eol=True)

            # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths

            best = numpy.argsort(costs)[0]
            try:
                total_cost += costs[best]
                trans_out = trans[best]
                weight = weights[best][:, :len(trans_in)]
                trans_out = self._idx_to_word(trans_out, self.trg_ivocab)
            except ValueError:
                logger.info(
                    "Can NOT find a translation for line: {}".format(i+1))
                trans_out = '<UNK>'

            saved_weights.append(weight)
            print(' '.join(trans_out), file=ftrans_original)
            pbar.update(i + 1)

        pbar.finish()
        ftrans_original.close()
        cPickle.dump(saved_weights, open(self.config['attention_weights'], 'wb'))
        self.data_stream.reset()

        # run afterprocess
        # self.ap.main()

        # calculate bleu
        bleu_subproc = Popen(self.compbleu_cmd, stdout=PIPE)
        while True:
            line = bleu_subproc.stdout.readline()
            if line != '':
                if 'BLEU' in line:
                    stdout = line
            else:
                break
        bleu_subproc.terminate()
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:]) * 100
        logger.info('BLEU: ' + str(bleu_score))
        self.main_loop.log.current_row['validation_bleu'] = bleu_score
        self.main_loop.log.current_row['validation_cost'] = total_cost

        return bleu_score, total_cost

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('score')).score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score, total_cost):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, 'bleu', self.config['saveto'])

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
            self.dump_parameters(self.main_loop, model.path)
            signal.signal(signal.SIGINT, s)

    def dump_parameters(self, main_loop, path):
        params_to_save = main_loop.model.get_parameter_values()
        param_values = {name.replace("/", "-"): param
                        for name, param in params_to_save.items()}
        outfile_path = path + '.' + str(main_loop.status['iterations_done'])
        with open(outfile_path, 'wb') as outfile:
            numpy.savez(outfile, **param_values)


class ModelInfo:
    """Utility class to keep track of evaluated models."""

    def __init__(self, score, name, path=None):
        self.score = score
        self.path = self._generate_path(path, name)

    def _generate_path(self, path, name):
        gen_path = os.path.join(
            path, name + '_%.2f' %
            (self.score) if path else None)
        return gen_path
