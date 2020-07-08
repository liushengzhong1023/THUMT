import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy
import re
from matplotlib import font_manager


def _open(filename, mode="r", encoding="utf-8"):
    if sys.version_info.major == 2:
        return open(filename, mode=mode)
    elif sys.version_info.major == 3:
        return open(filename, mode=mode, encoding=encoding)
    else:
        raise RuntimeError("Unknown Python version for running!")


def parse_numpy(string):
    string = string.replace('[', ' ').replace(']', ' ').replace(',', ' ')
    string = re.sub(' +', ' ', string)
    result = numpy.fromstring(string, sep=' ')
    return result

# # set font
# fontP = font_manager.FontProperties()
# fontP.set_family('SimHei')
# fontP.set_size(14)

# parse from text
local_result_path = "/Users/sliu/Desktop/Code/Python/THUMT/results/lrp_newstest2015"
result = _open(os.path.join(local_result_path, sys.argv[1]), 'r').read()
src = re.findall('src: (.*?)\n', result)[0]
# src = src.decode('utf-8')
trg = re.findall('trg: (.*?)\n', result)[0]
rlv = re.findall('result: ([\s\S]*)', result)[0]
rlv = parse_numpy(rlv)
src_words = src.split(' ')
src_words.append('<eos>')
trg_words = trg.split(' ')
trg_words.append('<eos>')

len_t = len(trg_words)
len_s = len(src_words)
rlv = rlv[:len_t*len_s]
rlv = numpy.reshape(rlv, [len_t, len_s])

# set the scale
maximum = numpy.max(numpy.abs(rlv))
plt.matshow(rlv, cmap="RdBu_r", vmin=-maximum, vmax=maximum)

# fontname = "Times"
plt.colorbar()
plt.xticks(range(len_s), src_words, fontsize=14,
           rotation='vertical')
plt.yticks(range(len_t), trg_words, fontsize=14)

matplotlib.rcParams['font.family'] = "Times"
# plt.show()

figure_path = "/Users/sliu/Desktop/Code/Python/THUMT/results/visualization"
basename = os.path.basename(sys.argv[1]).split('.')[0]
figure_file = os.path.join(figure_path, basename + '.pdf')
plt.savefig(figure_file)

