import re
import argparse
import configurations
from subprocess import Popen, PIPE

parser = argparse.ArgumentParser()
parser.add_argument(
    "--proto",  default="normal_adagrad",
    help="Prototype config to use for config")
args = parser.parse_args()

def main(config):
    compbleu_cmd = [config['bleu_script_1'], 
                         config['val_set_target'],
                         config['val_output_orig']]
    bleu_subproc = Popen(compbleu_cmd, stdout=PIPE)
    while True:
        line = bleu_subproc.stdout.readline()
        if line != '':
            if 'BLEU' in line:
                stdout = line
        else:
            break
    bleu_subproc.terminate()
    out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
    bleu_score = float(out_parse.group()[6:])
    print bleu_score

if __name__ == '__main__':
    config = getattr(configurations, args.proto)()
    main(config)
