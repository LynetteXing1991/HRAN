def hierarchical_s2sa_chinese_100w_withL():
    config = {}

    # model hyperparameters
    config['seq_len'] = 200
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620

    # traning options
    config['batch_size'] = 128
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['initial_learning_rate'] = 1.0
    config['learning_rate_decay'] = True
    config['learning_rate_grow'] = False
    config['step_clipping'] = 1.
    config['weight_scale'] = 0.01
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # data basics
    config['stream'] = 'stream'
    datadir = 'D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\\'
    config['datadir'] = datadir
    config['saveto'] = datadir+'HRAN_withNoL\\'
    config['validation_load']=datadir+'HRAN_withNoL\\models_for_test\\'
    config['model_name'] = 'HRAN_withNoL'
    # dictionary options
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'
    # dictionaries
    config['ctx_datas'] =[datadir+'100w.ctx_0.train',datadir+'100w.ctx_1.train',datadir+'100w.ctx_2.train'];
    config['val_ctx_datas'] =[datadir+'100w.ctx_0.test',datadir+'100w.ctx_1.test',datadir+'100w.ctx_2.test'];
    #config['val_ctx_datas'] =[datadir+'10w.ctx_0.test.ft',datadir+'10w.ctx_1.test.ft',datadir+'10w.ctx_2.test.ft'];
    config['ctx_num'] = 3
    config['src_data'] = datadir + '100w.query.train'
    config['trg_data'] = datadir + '100w.response.train'
    config['src_vocab'] = datadir + '100w.query.vocab.4w'
    config['trg_vocab'] = datadir + '100w.response.vocab.4w'
    config['src_vocab_size'] = 40000
    config['trg_vocab_size'] = 40000
    # # validation options
    config['val_set_source'] = datadir + '100w.query.test'
    config['val_set_target'] = datadir + '100w.response.test'
    # config['bleu_script'] = datadir + 'multi-bleu.perl'
    # config['bleu_script_1'] = datadir + 'CompBleu.exe'
    # config['normalized_bleu'] = True
    # testing options
    config['normalized_bleu'] = True
    # model saving related
    config['finish_after'] = 1000000
    config['reload'] = True
    config['save_freq'] = 500
    config['sampling_freq'] = 500
    config['hook_samples'] = 2
    config['bleu_val_freq'] = 1000
    config['val_burn_in'] = 20000

    # # afterprocess config
    # config['translation_table'] = datadir + 'translation_table'
    # config['replacement_table'] = datadir + 'test/' + 'replacement_table'

    config['use_doubly_stochastic'] = False
    config['lambda_ds'] = 0.001

    config['use_local_attention'] = False
    config['window_size'] = 10

    config['use_step_decay_cost'] = False

    config['use_concentration_cost'] = False
    config['lambda_ct'] = 10

    # arxiv.org/abs/1511.08400
    config['use_stablilizer'] = False
    config['lambda_st'] = 1

    config['match_function'] = 'SumMatchFunction'
    config['attention_images'] = config['saveto'] + '/attention_images/'
    config['attention_weights'] = config['validation_load'] + 'attention_weights'
    config['val_output_orig'] = config['validation_load'] + '/test_output_orig'
    config['val_output_repl'] = config['saveto'] + '/test_output_repl'

    config['beam_size']=10;

    return config