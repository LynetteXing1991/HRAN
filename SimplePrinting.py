import sys
from toolz import first
from blocks.extensions import SimpleExtension


class SimplePrinting(SimpleExtension):
    """Prints log messages to the screen."""
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        kwargs.setdefault("before_first_epoch", True)
        kwargs.setdefault("on_resumption", True)
        kwargs.setdefault("after_training", True)
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("on_interrupt", True)
        super(SimplePrinting, self).__init__(**kwargs)

    def _print_attributes(self, attribute_tuples):
        for attr, value in sorted(attribute_tuples.items(), key=first):
            if not attr.startswith("_"):
                print("\t" + "{}: {}".format(attr, value)),
        print
        sys.stdout.flush()

    def do(self, which_callback, *args):
        log = self.main_loop.log
        print_status = True

        # print()
        # print("".join(79 * "-"))
        if which_callback == "before_epoch" and log.status['epochs_done'] == 0:
            print("BEFORE FIRST EPOCH")
        elif which_callback == "on_resumption":
            print("TRAINING HAS BEEN RESUMED")
        elif which_callback == "after_training":
            print("TRAINING HAS BEEN FINISHED:")
        elif which_callback == "after_epoch":
            print("AFTER ANOTHER EPOCH")
        elif which_callback == "on_interrupt":
            print("TRAINING HAS BEEN INTERRUPTED")
            print_status = False
        # print("".join(79 * "-"))
        if print_status:
            # print("Training status:")
            # self._print_attributes(log.status)
            print self.model_name, log.status['iterations_done'],
            self._print_attributes(log.current_row)
        # print()
