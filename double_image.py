import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from progressbar import ProgressBar
from PIL import Image

dir1 = 'models/normal_adagrad/attention_images/'
dir2 = 'models/rec_adagrad/attention_images/'
outdir = 'double_images/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

candidates = range(20)

pbar = ProgressBar(max_value=len(candidates)).start()
for i, k in enumerate(candidates):
    pbar.update(i + 1)
    fig = plt.figure(figsize=(40, 20), dpi=80)
    f1 = dir1 + str(k) + '.png'
    f2 = dir2 + str(k) + '.png'
    for i, ff in enumerate([f2, f1]):
        image = Image.open(ff).convert('L')
        w, h = image.size
        image = image.crop((0, 0, w-500, h))
        arr = np.asarray(image)
        fig.add_subplot(1, 2, i)
        fig.tight_layout()
        plt.imshow(arr, cmap=cm.Blues)
        plt.tight_layout()
        plt.axis('off')
    plt.savefig(outdir + str(k) + '.png')
    plt.close()
pbar.finish()
