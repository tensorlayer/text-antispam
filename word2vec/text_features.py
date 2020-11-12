#! /usr/bin/env python3
"""生成用于CNN Classifier或者RNN Classifier的训练集。

训练好词向量后，将每一行文本转成词向量序列。
分别将正负样本保存到sample_seq_pass.npz和sample_seq_spam.npz。

"""

import numpy as np
import tensorlayer as tl
import sys
sys.path.append("../serving/packages")
from text_regularization import extractWords

wv = tl.files.load_npy_to_any(name='./output/model_word2vec_200.npy')
for label in ["pass", "spam"]:
    samples = []
    inp = "data/msglog/msg" + label + ".log.seg"
    outp = "output/sample_seq_" + label
    f = open(inp,encoding='utf-8')
    for line in f:
        line = extractWords(line)
        words = line.strip().split(' ')
        text_sequence = []
        for word in words:
            try:
                text_sequence.append(wv[word])
            except KeyError:
                text_sequence.append(wv['UNK'])
        samples.append(text_sequence)

    if label == "spam":
        labels = np.zeros(len(samples))
    elif label == "pass":
        labels = np.ones(len(samples))

    np.savez(outp, x=samples, y=labels)
    f.close()
