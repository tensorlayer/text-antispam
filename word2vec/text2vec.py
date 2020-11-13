#! /usr/bin/env python3
"""生成用于NBOW+MLP Classifier的训练集。

训练好词向量后，通过将词向量线性相加获得文本的向量。
输入分词后的样本，每一行逐词查找词向量并相加，从而得到文本的特征向量和标签。
分别将正负样本保存到sample_pass.npz和sample_spam.npz。

"""

import numpy as np
import tensorlayer as tl
import sys
sys.path.append("../serving/packages")
from text_regularization import extractWords

wv = tl.files.load_npy_to_any(name='./output/model_word2vec_200.npy')
for label in ["pass", "spam"]:
    embeddings = []
    inp = "data/msglog/msg" + label + ".log.seg"
    outp = "output/sample_" + label
    f = open(inp, encoding='utf-8')
    for line in f:
        line = extractWords(line)
        words = line.strip().split(' ')
        text_embedding = np.zeros(200)
        for word in words:
            try:
                text_embedding += wv[word]
            except KeyError:
                text_embedding += wv['UNK']
        embeddings.append(text_embedding)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if label == "spam":
        labels = np.zeros(embeddings.shape[0])
    elif label == "pass":
        labels = np.ones(embeddings.shape[0])

    np.savez(outp, x=embeddings, y=labels)
    f.close()
