import os
import numpy
import logging
import operator
import cPickle
from blocks.extensions.training import SharedVariableModifier, SimpleExtension
from blocks.serialization import secure_dump, load, BRICK_DELIMITER
from checkpoint import SaveLoadUtils

BLEU = 'validation_bleu'
COST = 'validation_cost'

logger = logging.getLogger(__name__)


def halver(t, x):
    return x / 2


class LearningRateHalver(SharedVariableModifier, SaveLoadUtils):

    def __init__(self, record_name, comparator, learning_rate, 
                 patience_default, lower_threshold=0.001):
        self.record_name = record_name
        self.comparator = comparator
        self.learning_rate = learning_rate
        self.patience_default = patience_default
        self.patience = patience_default
        self.lower_threshold = lower_threshold
        super(LearningRateHalver, self).__init__(self.learning_rate, halver)

    def do_half_nan(self):
        if 'perplexity' in self.main_loop.log.current_row:
            pepl = self.main_loop.log.current_row['perplexity'].tolist()
            return numpy.isnan(pepl)

    def do_half_patient(self):
        logs = sorted(self.main_loop.log.items(), 
                      key=operator.itemgetter(0), 
                      reverse=True)
        bleu_values = [y[self.record_name] for x, y in logs 
                       if self.record_name in y]
        if len(bleu_values) < 2:
            return False
        current_value = bleu_values[-1]
        previous_value = bleu_values[-2]
        if self.comparator(current_value, previous_value):
            self.patience -= 1
            if self.patience == 0:
                self.patience = self.patience_default
                return True
        else:
            self.patience = self.patience_default
            self.remove_old_models()
        return False

    def reload_parameters(self, path):
        params = self.load_parameter_values(path)
        self.set_model_parameters(self.main_loop.model, params)

    def reload_iteration_state(self, path):
        with open(path, 'rb') as source:
            self.main_loop.iteration_state = load(source)

    def reload_log(self, path):
        with open(path, 'rb') as source:
            self.main_loop.log = cPickle.load(source)
    
    def reload_previous_model(self, step_back):
        paths = sorted(self.main_loop.log.items(), 
                       key=operator.itemgetter(0),
                       reverse=True)
        paths = [y['saved_to'] for x, y in paths if 'saved_to' in y]
        paths = [path for path in paths if all([os.path.exists(p) for p in path])]
        if len(paths) < 1:
            return
        idx = min(step_back, len(paths) - 1)
        path = paths[idx]
        to_be_removed = paths[:idx] + paths[idx+1:]
        reload_from = path[0].split('.')[-1]
        logger.info('Reloading model from ' + reload_from)
        self.reload_parameters(path[0])
        self.reload_iteration_state(path[1])
        self.reload_log(path[2])
        self.main_loop.log.current_row['reload_from'] = int(reload_from)
        self.remove_models(to_be_removed)

    def remove_models(self, paths):
        [os.remove(p) for pp in paths for p in pp if os.path.exists(p)]

    def remove_old_models(self):
        paths = sorted(self.main_loop.log.items(), 
                       key=operator.itemgetter(0),
                       reverse=True)
        paths = [y['saved_to'] for x, y in paths if 'saved_to' in y]
        paths = [path for path in paths if all([os.path.exists(p) for p in path])]
        to_be_removed = paths[:-3]
        self.remove_models(to_be_removed)

    def do(self, which_callback, *args):
        current_learning_rate = self.learning_rate.get_value().tolist()
        self.main_loop.log.current_row['learning_rate'] = current_learning_rate
        if current_learning_rate < self.lower_threshold:
            self.main_loop.log.current_row['training_finish_requested'] = True
        if self.record_name in self.main_loop.log.current_row:
            if self.do_half_nan():
                self.reload_previous_model(1)
                super(LearningRateHalver, self).do(which_callback, *args)
            if self.do_half_patient():
                self.reload_previous_model(self.patience_default)
                super(LearningRateHalver, self).do(which_callback, *args)


def doubler(t, x):
    return x / 2


class LearningRateDoubler(SharedVariableModifier):

    def __init__(self, record_name, comparator, learning_rate, patience_default):
        self.record_name = record_name
        self.comparator = comparator
        self.learning_rate = learning_rate
        self.patience_default = patience_default
        self.patience = patience_default
        super(LearningRateDoubler, self).__init__(self.learning_rate, doubler)

    def do_double(self):
        logs = sorted(self.main_loop.log.items(), 
                      key=operator.itemgetter(0), 
                      reverse=True)
        bleu_values = [y[self.record_name] for x, y in logs 
                       if self.record_name in y]
        if len(bleu_values) < 2:
            return False
        current_value = bleu_values[-1]
        previous_value = bleu_values[-2]
        if self.comparator(current_value, previous_value):
            self.patience -= 1
            if self.patience == 0:
                self.patience = self.patience_default
                return True
        else:
            self.patience = self.patience_default
        return False

    def do(self, which_callback, *args):
        self.main_loop.log.current_row['learning_rate'] = \
            self.learning_rate.get_value().tolist()
        if self.record_name in self.main_loop.log.current_row:
            if self.do_double():
                super(LearningRateDoubler, self).do(which_callback, *args)


class OldModelRemover(SimpleExtension):

    def __init__(self, saveto, **kwargs):
        self.saveto = saveto
        super(OldModelRemover, self).__init__(**kwargs)

    def remove_old_models(self):
        params_prefix = 'params.npz.'
        states_prefix = 'iteration_states.pkl.'
        logs_prefix = 'log.'
        fnames = os.listdir(self.saveto)
        params = [f for f in fnames if f.startswith(params_prefix)]
        states = [f for f in fnames if f.startswith(states_prefix)]
        logs = [f for f in fnames if f.startswith(log_prefix)]
        for f in params + states + logs:
            num = int(f.split('.')[-1])
            if self.main_loop.status['iterations_done'] - num > 3000:
                os.remove(os.path.join(self.saveto, f))

    def do(self, which_callback, *args):
        current_row = self.main_loop.log.current_row
        if BLEU in current_row or COST in current_row:
            remove_old_models()
