#coding:utf-8

import os
import time
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from progressbar import ProgressBar
from PIL import Image
import sys
import configurations_base
import argparse
import logging
import pprint


parser = argparse.ArgumentParser()
parser.add_argument(
    "--proto",  default="normal_adagrad",
    help="Prototype config to use for config")
args = parser.parse_args()


def showmat(name, mat, alpha, beta):
    sample_dir = 'samples/'
    alpha = [a.decode('utf-8') for a in alpha]
    beta = [b.decode('utf-8') for b in beta]

    fig = plt.figure(figsize=(20, 20), dpi=80)
    plt.clf()
    matplotlib.rcParams.update({'font.size': 12})
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.xaxis.tick_top()
    res = ax.imshow(mat, cmap=plt.cm.Blues,
                    interpolation='nearest')
    
    font_prop = FontProperties()
    font_prop.set_file('D:\users\chxing\\aaai2017Exp\\filterDouban\data.r50.q1min30max.tokenized.2-20.1-20\model_for_test\\attention\\simfang.ttf')
    font_prop.set_size('large')
    
    plt.xticks(range(len(alpha)), alpha, rotation=60, 
               fontproperties=font_prop)
    plt.yticks(range(len(beta)), beta, fontproperties=font_prop)

    cax = plt.axes([0.0, 0.0, 0.0, 0.0])
    plt.colorbar(mappable=res, cax=cax)
    plt.savefig(name + '.png', format='png')
    plt.close()


def main(config):
    images_dir = config['attention_images']
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    source_file = open(config['test_set_source'], 'r').readlines()
    #target_file = pickle.load(open(config['val_output_repl'] + '.pkl', 'rb'))
    target_file = open(config['val_output_orig'], 'r').readlines()
    weights = pickle.load(open(config['attention_weights'], 'rb'))

    pbar = ProgressBar(max_value=len(source_file)).start()
    for i, (ctx_0,ctx_1,ctx_2,source, target, weight) in enumerate(
            zip( ctx_files[0],ctx_files[1],ctx_files[2],source_file, target_file, weights)):
        pbar.update(i + 1)
        lenList=[len(source.strip().split()),len(ctx_0.strip().split()),len(ctx_1.strip().split()),len(ctx_2.strip().split())]
        maxLen=max(lenList);
        sources = source.strip().split()+['EOS']+['pad' for j in range(maxLen-lenList[0])]+\
                  ctx_0.strip().split()+['EOS']+['pad' for j in range(maxLen-lenList[1])]\
                  +ctx_1.strip().split()+['EOS']+['pad' for j in range(maxLen-lenList[2])]\
                  +ctx_2.strip().split()+['EOS']+['pad' for j in range(maxLen-lenList[3])];
        target=target.strip().split()
        showmat(images_dir + str(i), weight, sources, target)
    pbar.finish()

def main_forTest(config):
    images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    weights = pickle.load(open(config['attention_weights'], 'rb'))
    weight=weights[4620];
    #final_weight=weight[:,14:35].extend(weight[:,42:49]);#+weight[:,9]
    final_weight=np.concatenate((weight[:,14:35],weight[:,42:49]),axis=1);
    final_weight=np.concatenate((final_weight,weight[:,:9]),axis=1);
    final_weight2=np.delete(final_weight,[11,12,30,31],1);
    final_weight2[:,5]+=final_weight[:,11];
    final_weight2[:,5]+=final_weight[:,12];
    final_weight2[:,29]+=final_weight[:,30];
    final_weight2[:,30]+=final_weight[:,31];
    final_weight3=np.delete(final_weight2,[11,18,25,32],1)
    final_weight3[:,9]+=final_weight2[:,11];
    final_weight3[:,12]+=final_weight2[:,18]/2;
    final_weight3[:,16]+=final_weight2[:,18]/2;
    final_weight3[:,22]+=final_weight2[:,25];
    final_weight3[:,28]+=final_weight2[:,32];
    # ctx_0="征 男 票 160cm 的 妹子 真的 找不到 男 票 吗";
    # ctx_1="你 找不到 一定 不是 因为 160";
    # ctx_2="我 知道 脸 也是 硬伤 嘛";
    # source="是 你 非 要 175 以上";
    # target="身高 不是 硬性 要求 </S>";
    final_weight3=np.sum(final_weight3,axis=0,keepdims=1);
    final_weight3/=5;
    final_weight3=final_weight3[:,23:29]
    # final_weight4=np.delete(final_weight3,[2,9],1);
    # final_weight4[:,7]+=final_weight3[:,2];
    # final_weight4[:,7]+=final_weight3[:,9];
    # sources = ctx_0.strip().split()+ctx_1.strip().split()+ctx_2.strip().split()+source.strip().split();
    # target=target.strip().split()
    sources=[' '];
    target=[' '];
    showmat(images_dir+str(4620)+'_m', final_weight3, sources, target)

def main_forTest_4535_u(config):
    images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    weights = pickle.load(open(config['attention_weights'], 'rb'))
    weight=weights[4535];
    final_weight2=np.concatenate((weight[:,11:43],weight[:,:7]),axis=1);
    final_weight3=[final_weight2[:,32]+final_weight2[:,33]+final_weight2[:,34],final_weight2[:,35],final_weight2[:,36],final_weight2[:,37]+final_weight2[:,38]];
    final_weight3=np.sum(final_weight3,axis=1,keepdims=1);
    #source="u1 u2 u3 u4";
    #target="你 用 的 什么 牌子 的 化妆水";
    #sources = source.strip().split();
    #target=target.strip().split()
    sources=[' '];
    target=[' '];
    showmat(images_dir+str(4535)+'_u', final_weight3, sources, target)

def main_forTest_4535(config):
    images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    weights = pickle.load(open(config['attention_weights'], 'rb'))
    weight=weights[4535];
    #final_weight=weight[:,14:35].extend(weight[:,42:49]);#+weight[:,9]
    final_weight=np.concatenate((weight[:,11:43],weight[:,:7]),axis=1);
    final_weight2=np.delete(final_weight,[10,19,20,21,31,32,33,38],1);
    final_weight2[:,7]+=final_weight[:,10];
    final_weight2[:,14]+=final_weight[:,20];
    final_weight2[:,15]+=final_weight[:,21]+final_weight[:,19];
    final_weight2[:,17]+=final_weight[:,31];
    final_weight2[:,27]+=final_weight[:,33]+final_weight[:,32]+final_weight[:,38];
    final_weight2=np.sum(final_weight2,axis=0,keepdims=1);
    final_weight2/=6;
    final_weight2=final_weight2[:,26:31]

    # ctx_0="我 不能 去 你 可以 找 丽丽 陪 你 吃饭";
    # ctx_1="她 住 松江 离 市区 太远 了";
    # ctx_2="有 好吃 的 多 远 都 要 去 啊";
    # source="你 为什么 不能 来 呢";
    # target="吃 多 了 上火 了 </S>";
    # sources = ctx_0.strip().split()+ctx_1.strip().split()+ctx_2.strip().split()+source.strip().split();
    # target=target.strip().split()
    sources=[' '];
    target=[' '];
    showmat(images_dir+str(4535)+'_m', final_weight2, sources, target)

def main_forTest_4408_u(config):
    images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    weights = pickle.load(open(config['attention_weights'], 'rb'))
    weight=weights[4408];
    final_weight=np.concatenate((weight[:,9:16],weight[:,18:22]),axis=1);
    final_weight=np.concatenate((final_weight,weight[:,27:34]),axis=1);
    final_weight=np.concatenate((final_weight,weight[:,0:9]),axis=1);
    final_weight2=np.delete(final_weight,[6,10,17,26],1);
    final_weight2=final_weight2[:6,:];
    final_weight3=[final_weight2[:,15]+final_weight2[:,16]+final_weight2[:,17],final_weight2[:,18]+final_weight2[:,19],final_weight2[:,20]+final_weight2[:,21],final_weight2[:,22]];
    final_weight3=np.sum(final_weight3,axis=1,keepdims=1);
    final_weight3/=6
    source="u1 u2 u3 u4";
    target="那 你 做 过 设计 么";
    sources = [' '];
    target= [' '];
    showmat(images_dir+str(4408)+'_u', final_weight3, sources, target)

def main_forTest_4408(config):
    images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    weights = pickle.load(open(config['attention_weights'], 'rb'))
    weight=weights[4408];
    #final_weight=weight[:,14:35].extend(weight[:,42:49]);#+weight[:,9]
    final_weight=np.concatenate((weight[:,9:16],weight[:,18:22]),axis=1);
    final_weight=np.concatenate((final_weight,weight[:,27:34]),axis=1);
    final_weight=np.concatenate((final_weight,weight[:,0:9]),axis=1);
    final_weight2=np.delete(final_weight,[6,10,17,26],1);
    final_weight2[:,5]+=final_weight[:,6];
    final_weight2[:,8]+=final_weight[:,10];
    final_weight2[:,14]+=final_weight[:,17];
    final_weight2[:,22]+=final_weight[:,26];
    final_weight2=np.sum(final_weight2,axis=0,keepdims=1);
    final_weight2/=6;
    final_weight2=final_weight2[:,15:23]
    # ctx_0="啦啦 啦啦 销售助理 在哪里 呀 在哪里";
    # ctx_1="学历 不 达标";
    # ctx_2="对 做 销售 有 兴趣 么";
    # source="没 做 过 不过 我 不 挑 工作";
    # target="那 你 做 过 设计 么";
    # sources = ctx_0.strip().split()+ctx_1.strip().split()+ctx_2.strip().split()+source.strip().split();
    # target=target.strip().split()
    sources=[' '];
    target=[' '];
    showmat(images_dir+str(4408)+'_m', final_weight2, sources, target)

# def main_forTest_4408(config):
#     images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";
#
#     ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
#     weights = pickle.load(open(config['attention_weights'], 'rb'))
#     weight=weights[4408];
#     #final_weight=weight[:,14:35].extend(weight[:,42:49]);#+weight[:,9]
#     final_weight=np.concatenate((weight[:,9:16],weight[:,18:22]),axis=1);
#     final_weight=np.concatenate((final_weight,weight[:,27:34]),axis=1);
#     final_weight=np.concatenate((final_weight,weight[:,0:9]),axis=1);
#     final_weight2=np.delete(final_weight,[6,10,17,26],1);
#     final_weight2[:,5]+=final_weight[:,6];
#     final_weight2[:,8]+=final_weight[:,10];
#     final_weight2[:,14]+=final_weight[:,17];
#     final_weight2[:,22]+=final_weight[:,26];
#     # final_weight3=np.delete(final_weight2,[11,18,25,32],1)
#     # final_weight3[:,9]+=final_weight2[:,11];
#     # final_weight3[:,12]+=final_weight2[:,18]/2;
#     # final_weight3[:,16]+=final_weight2[:,18]/2;
#     # final_weight3[:,22]+=final_weight2[:,25];
#     # final_weight3[:,28]+=final_weight2[:,32];
#     ctx_0="啦啦 啦啦 销售助理 在哪里 呀 在哪里";
#     ctx_1="学历 不 达标";
#     ctx_2="对 做 销售 有 兴趣 么";
#     source="没 做 过 不过 我 不 挑 工作";
#     target="那 你 做 过 设计 么";
#     sources = ctx_0.strip().split()+ctx_1.strip().split()+ctx_2.strip().split()+source.strip().split();
#     target=target.strip().split()
#     showmat(images_dir+str(4408)+'_1', final_weight2[:6,:], sources, target)

def main_forTest_4860(config):
    images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    weights = pickle.load(open(config['attention_weights'], 'rb'))
    weight=weights[4860];
    #final_weight=weight[:,14:35].extend(weight[:,42:49]);#+weight[:,9]
    final_weight=np.concatenate((weight[:,13:21],weight[:,26:44]),axis=1);
    final_weight=np.concatenate((final_weight,weight[:,:6]),axis=1);
    final_weight2=np.delete(final_weight,[7,20,25,31],1);
    final_weight2[:,3]+=final_weight[:,7];
    final_weight2[:,17]+=final_weight[:,20];
    final_weight2[:,22]+=final_weight[:,25];
    final_weight2[:,27]+=final_weight[:,31];
    final_weight3=np.delete(final_weight2,[8,9],1);
    final_weight3[:,13]+=final_weight2[:,8];
    final_weight3[:,13]+=final_weight2[:,9];
    final_weight3=np.sum(final_weight3,axis=0,keepdims=1);
    final_weight3/=7;
    final_weight3=final_weight3[:,21:26]
    # final_weight3=np.delete(final_weight2,[11,18,25,32],1)
    # final_weight3[:,9]+=final_weight2[:,11];
    # final_weight3[:,12]+=final_weight2[:,18]/2;
    # final_weight3[:,16]+=final_weight2[:,18]/2;
    # final_weight3[:,22]+=final_weight2[:,25];
    # final_weight3[:,28]+=final_weight2[:,32];
    ctx_0="求 唇 部 死 皮 怎么 去";
    # ctx_1="用 化妆水 一遍 遍 的 擦 嘴唇 就能 擦 下来";
    # ctx_2="好 的 回去 试试";
    # source="我 是 能 擦 下来";
    # target="你 用 的 什么 牌子 的 化妆水";
    sources = [' '];
    #target=target.strip().split()
    target=[' '];
    showmat(images_dir+str(4860)+'_m', final_weight3, sources, target)

def main_forTest_4860_u(config):
    images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    weights = pickle.load(open(config['attention_weights'], 'rb'))
    weight=weights[4860];
    final_weight=np.concatenate((weight[:,13:21],weight[:,26:44]),axis=1);
    final_weight2=final_weight[:,:7];
    final_weight3=[final_weight2[:,0],final_weight2[:,1]+final_weight2[:,5],final_weight2[:,4]+final_weight2[:,3]+final_weight2[:,2],final_weight2[:,6]];
    final_weight3=np.sum(final_weight3,axis=1,keepdims=1);
    #source="u1 u2 u3 u4";
    #target="你 用 的 什么 牌子 的 化妆水";
    #sources = source.strip().split();
    #target=target.strip().split()
    sources=[' '];
    target=[' '];
    showmat(images_dir+str(4860)+'_u', final_weight3, sources, target)


def main_forTest_u(config):
    images_dir = "D:\users\chxing\\acl2017Exp\data\\filteredDouban\\100w\semi_structured_model\models_for_test\\picForPaper\\";

    ctx_files=[open(config['test_ctx_datas'][i]).readlines() for i in range(config['ctx_num'])]
    weights = pickle.load(open(config['attention_weights'], 'rb'))
    weight=weights[4620];
    final_weight=weight[:,:9];
    final_weight2=[final_weight[:,1]+final_weight[:,0]+final_weight[:,2]+final_weight[:,3]+final_weight[:,6],final_weight[:,5],final_weight[:,4],final_weight[:,7]+final_weight[:,8]];
    final_weight2=np.sum(final_weight2,axis=1,keepdims=1);
    final_weight2/=5;
    # source="u1 u2 u3 u4";
    # target="身高 不是 硬性 要求 </S>";
    # sources = source.strip().split();
    # target=target.strip().split()
    sources=[' '];
    target=[' '];
    showmat(images_dir+str(4620)+'_u', final_weight2, sources, target)

def crop(config):
    indir = config['attention_images']
    outdir = config['attention_images'] + '/cropped'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for fname in os.listdir(indir):
        inpath = os.path.join(indir, fname)
        outpath = os.path.join(outdir, fname)
        if os.path.isdir(inpath):
            continue
        image = Image.open(inpath)
        w, h = image.size
        image = image.crop((0, 0, w, h-12))
        image.save(outpath, 'png')

logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--proto",  default="hierarchical_s2sa_chinese_100w_posTag",
    help="Prototype config to use for config")
args = parser.parse_args()


if __name__ == '__main__':
    config = getattr(configurations_base, args.proto)()
    main_forTest(config)
    crop(config)
