import cPickle
import numpy as np
import configurations
import argparse
import operator

class afterprocesser:

    def __init__(self, config):
        self.config = config

    def is_unk(self, s):
        return s == '<UNK>'

    def is_dollar(self, s):
        return s.startswith('$')
    
    def is_eol(self, s):
        return s == '</S>'
    
    def process_sent(self, src, sent, weights, 
                     trans_table, repl_table, att_table):
        replaced = []
        replaced_cut = []
        for word, ws in zip(sent, weights):
            att = np.argmax(ws)
            if self.is_unk(word):
                if att in repl_table:
                    mark, repl = repl_table[att]
                    replaced.append(repl)
                    replaced_cut.append(repl)
                elif att < len(src):
                    if src[att] in trans_table:
                        repl, freq = trans_table[src[att]][0]
                        replaced.append(repl)
                        replaced_cut.append(repl)
                    elif src[att] in att_table:
                        repl, freq = att_table[src[att]][0]
                        replaced.append(repl)
                        replaced_cut.append(repl)
                    else:
                        replaced.append('$' + src[att].strip())
                else:
                    replaced.append(word)
            elif self.is_dollar(word):
                if att in repl_table:
                    mark, repl = repl_table[att]
                    replaced.append(repl)
                    replaced_cut.append(repl)
                elif att < len(src):
                    if src[att] in trans_table:
                        repl, freq = trans_table[src[att]][0]
                        replaced.append(repl)
                        replaced_cut.append(repl)
                    elif src[att] in att_table:
                        repl, freq = att_table[src[att]][0]
                        replaced.append(repl)
                        replaced_cut.append(repl)
                    else:
                        replaced.append('$' + src[att].strip())
                else:
                    replaced.append(word)
            elif not self.is_eol(word):
                replaced.append(word)
                replaced_cut.append(word)
        return replaced, replaced_cut
    
    
    def main(self):
        val_set = self.config['val_set_source']
        source_file = open(val_set, 'r').readlines()
        original_file = open(self.config['val_output_orig'], 'r').readlines()
        replaced_file = open(self.config['val_output_repl'], 'wb')
        replaced_pkl = open(self.config['val_output_repl'] + '.pkl', 'wb')
        weights = cPickle.load(open(self.config['attention_weights'], 'rb'))
        translation_table = cPickle.load(open(self.config['translation_table'], 'rb'))
        replacement_table = cPickle.load(open(self.config['replacement_table'], 'rb'))

        att_table = dict()
        '''
        for mat, src, trg in zip(weights, source_file, original_file):
            src = src.split()
            trg = trg.split()
            for line, word in zip(mat.T, src):
                line = line / line.sum()
                i = line.argmax()
                if self.is_unk(trg[i]) or self.is_dollar(trg[i]) or self.is_eol(trg[i]):
                    continue
                if word not in att_table:
                    att_table[word] = dict()
                if trg[i] not in att_table[word]:
                    att_table[word][trg[i]] = 0
                att_table[word][trg[i]] += 1
        for key, value in att_table.items():
            value = sorted(value.items(), key=operator.itemgetter(1), reverse=True)
            att_table[key] = value
        '''
    
        all_replaced = []
        for i, (source, sent, weight, repl) in enumerate(zip(
                source_file, original_file, weights, replacement_table)):
            sent = sent.strip().split(' ')
            source = source.strip().split(' ')
            replaced, replaced_cut = self.process_sent(source, sent, weight,
                                                  translation_table, repl,
                                                  att_table)
            all_replaced.append(replaced)
            replaced_file.write(' '.join(replaced_cut) + '\n')
        cPickle.dump(all_replaced, replaced_pkl)


if __name__ == '__main__':
    config = configurations.normal_adagrad()
    ap = afterprocesser(config)
    ap.main()
