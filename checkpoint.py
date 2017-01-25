import logging
import numpy
import os
import time

from contextlib import closing
from six.moves import cPickle

from blocks.extensions.saveload import SAVED_TO, LOADED_FROM
from blocks.extensions import TrainingExtension, SimpleExtension
from blocks.serialization import secure_dump, load, BRICK_DELIMITER

logger = logging.getLogger(__name__)


class SaveLoadUtils(object):

    @property
    def path_to_folder(self):
        return self.folder

    @property
    def path_to_parameters(self):
        return os.path.join(self.folder, 'params.npz')

    @property
    def path_to_iter_state(self):
        return os.path.join(self.folder, 'iterations_state.pkl')

    @property
    def path_to_log(self):
        return os.path.join(self.folder, 'log')

    def load_parameter_values(self, path):
        with closing(numpy.load(path)) as source:
            param_values = {}
            for name, value in source.items():
                if name != 'pkl':
                    name_ = name.replace(BRICK_DELIMITER, '/')
                    if not name_.startswith('/'):
                        name_ = '/' + name_
                    param_values[name_] = value
        return param_values

    def save_parameter_values(self, param_values, path):
        param_values = {name.replace("/", "-"): param
                        for name, param in param_values.items()}
        with open(path, 'wb') as outfile:
            numpy.savez(outfile, **param_values)

    def set_model_parameters(self, model, params):
        params_this = model.get_parameter_dict()
        missing = set(params_this.keys()) - set(params.keys())
        for pname in params_this.keys():
            if pname in params:
                val = params[pname]
                if params_this[pname].get_value().shape != val.shape:
                    logger.warning(
                        " Dimension mismatch {}-{} for {}"
                        .format(params_this[pname].get_value().shape,
                                val.shape, pname))

                params_this[pname].set_value(val)
                '''
                logger.info(" Loaded to CG {:15}: {}"
                            .format(val.shape, pname))
                '''
            else:
                logger.warning(
                    " Parameter does not exist: {}".format(pname))
        logger.info(
            " Number of parameters loaded for computation graph: {}"
            .format(len(params_this) - len(missing)))


class CheckpointNMT(SimpleExtension, SaveLoadUtils):

    def __init__(self, saveto, model_name, **kwargs):
        self.folder = saveto
        self.model_name = model_name
        kwargs.setdefault("after_training", True)
        super(CheckpointNMT, self).__init__(**kwargs)

    def enhance_path(self, main_loop, path):
        return path + '.' + str(main_loop.status['iterations_done'])

    def dump_parameters(self, main_loop):
        params_to_save = main_loop.model.get_parameter_values()
        self.save_parameter_values(params_to_save,
                                   self.enhance_path(main_loop,
                                                     self.path_to_parameters))

    def dump_iteration_state(self, main_loop):
        secure_dump(main_loop.iteration_state,
                    self.enhance_path(main_loop, self.path_to_iter_state))

    def dump_log(self, main_loop):
        secure_dump(main_loop.log,
                    self.enhance_path(main_loop, self.path_to_log),
                    cPickle.dump)

    def dump(self, main_loop):
        if not os.path.exists(self.path_to_folder):
            os.mkdir(self.path_to_folder)
        print("")
        logger.info(" Saving model: " + self.model_name)
        start = time.time()
        logger.info(" Saving parameters")
        self.dump_parameters(main_loop)
        logger.info(" Saving iteration state")
        self.dump_iteration_state(main_loop)
        logger.info(" Saving log")
        self.dump_log(main_loop)
        logger.info(" Model saved, took {} seconds.".format(time.time()-start))

    def do(self, callback_name, *args):
        try:
            self.dump(self.main_loop)
        except Exception:
            raise
        finally:
            '''
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (self.path_to_folder +
                                                            'params.npz',))
            '''
            self.main_loop.log.current_row[SAVED_TO] = [
                self.enhance_path(self.main_loop, self.path_to_parameters),
                self.enhance_path(self.main_loop, self.path_to_iter_state),
                self.enhance_path(self.main_loop, self.path_to_log)]


class LoadNMT(TrainingExtension, SaveLoadUtils):

    def __init__(self, saveto, **kwargs):
        self.folder = saveto
        super(LoadNMT, self).__init__(saveto, **kwargs)

    def before_training(self):
        if not os.path.exists(self.path_to_folder):
            return
        self.load_last_model(self.main_loop)

    def load_parameters(self, path):
        return self.load_parameter_values(path)

    def load_parameters_default(self):
        return self.load_parameter_values(self.path_to_parameters)

    def load_iteration_state(self, path):
        with open(path, "rb") as source:
            return load(source)

    def load_log(self, path):
        with open(path, "rb") as source:
            return cPickle.load(source)

    def get_last_save(self, saves, prefix):
        if len(saves) == 0:
            return None
        if prefix in saves:
            return prefix
        nums = [int(s[len(prefix)+1:]) for s in saves]
        return prefix + '.' + str(max(nums))

    def load_last_model(self, main_loop):
        param_prefix = 'params.npz'
        state_prefix = 'iterations_state.pkl'
        log_prefix = 'log'

        files = os.listdir(self.folder)
        params = [f for f in files if f.startswith(param_prefix)]
        states = [f for f in files if f.startswith(state_prefix)]
        logs = [f for f in files if f.startswith(log_prefix)]

        param_name = self.get_last_save(params, param_prefix)
        if param_name is not None:
            logger.info(" Loading params from " + param_name)
            params_all = self.load_parameters(
                os.path.join(self.path_to_folder, param_name))
            self.set_model_parameters(main_loop.model, params_all)

        state_name = self.get_last_save(states, state_prefix)
        if state_name is not None:
            logger.info(" Loading state from " + state_name)
            main_loop.iteration_state = self.load_iteration_state(
                os.path.join(self.path_to_folder, state_name))

        log_name = self.get_last_save(logs, log_prefix)
        if log_name is not None:
            logger.info(" Loading log from " + log_name)
            main_loop.log = self.load_log(
                os.path.join(self.path_to_folder, log_name))

        if param_name is not None and len(param_name) < len(param_prefix):
            main_loop.log.current_row[LOADED_FROM] = \
                int(param_name[len(param_prefix)+1:])
