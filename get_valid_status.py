import os
import sys
import cPickle
import operator
import matplotlib.pyplot as plt


model_dir = 'models/'
BLEU = 'validation_bleu'
COST = 'validation_cost'
PLOT_BLEU = True
PLOT_COST = False


def get_log(path):
    filenames = os.listdir(path)
    logs = [f for f in filenames if f.startswith('log')]
    if len(logs) == 0:
        return None
    iterations = [int(l.split('.')[-1]) for l in logs if '.' in l]
    if len(iterations) == 0:
        return cPickle.load(open(os.path.join(path, logs[0]), 'rb'))
    x = max(iterations)
    return cPickle.load(open(os.path.join(path, 'log.' + str(x)), 'rb'))


lines = []
names = []
def main():
    for model_name in os.listdir(model_dir):
        if model_name.endswith('bk'):
            continue
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        log = get_log(model_path)
        if log is None:
            continue
        log = sorted(log.items(), key=operator.itemgetter(0))
        bleus = [(x, y[BLEU]) for x, y in log if BLEU in y]
        costs = [(x, y[COST]) for x, y in log if COST in y]
        if len(bleus) == 0:
            continue

        if PLOT_BLEU:
            line, = plt.plot(*zip(*bleus), 
                             linewidth=2.0, 
                             label=model_name + ' bleu', 
                             marker='+')
            names.append(model_name + ' bleu')

        if PLOT_COST:
            line, = plt.plot(*zip(*costs), 
                             linewidth=2.0, 
                             label=model_name + 'cost', 
                             marker='+')
            names.append(model_name + ' cost')
        lines.append(line)


    plt.legend(lines, names, loc=4)
    plt.show()


if __name__ == '__main__':
    main()
