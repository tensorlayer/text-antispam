#! /usr/bin/env python3
"""Word2vec训练词向量

Efficient Estimation of Word Representations in Vector Space: https://arxiv.org/pdf/1301.3781.pdf
Distributed Representations of Words and Phrases and their Compositionality: https://arxiv.org/pdf/1310.4546.pdf
word2vec Parameter Learning Explained: https://arxiv.org/pdf/1411.2738.pdf

"""

import collections
import logging
import os
import tarfile
import tensorflow as tf
import tensorlayer as tl


def load_dataset():
    """加载训练数据
    Args:
        files: 词向量训练数据集合
            得 我 就 在 车里 咪 一会
            终于 知道 哪里 感觉 不 对 了
            ...
    Returns:
        [得 我 就 在 车里 咪 一会 终于 知道 哪里 感觉 不 对 了...]
    """
    if not os.path.exists('data/postlog'):
        tl.files.maybe_download_and_extract(
            'postlog.tar.gz',
            'data',
            'https://github.com/pakrchen/text-antispam/raw/master/word2vec/data/')
        tarfile.open('data/postlog.tar.gz', 'r').extractall('data')
    files = ['data/postlog/postpass.log.seg', 'data/postlog/postspam.log.seg']
    words = []
    for file in files:
        f = open(file)
        for line in f:
            for word in line.strip().split(' '):
                if word != '':
                    words.append(word)
        f.close()
    return words


def get_vocabulary_size(words, min_freq=3):
    """获取词频不小于min_freq的单词数量
    小于min_freq的单词统一用UNK（unknown）表示
    Args:
        words: 训练词表
            [得 我 就 在 车里 咪 一会 终于 知道 哪里 感觉 不 对 了...]
        min_freq: 最低词频
    Return:
        size: 词频不小于min_freq的单词数量
    """
    size = 1 # 为UNK预留
    counts = collections.Counter(words).most_common()
    for word, c in counts:
        if c >= min_freq:
            size += 1
    return size


def save_checkpoint(ckpt_file_path):
    """保存模型训练状态
    将会产生以下文件:
        checkpoint
        model_name.ckpt.data-?????-of-?????
        model_name.ckpt.index
        model_name.ckpt.meta
    Args:
        ckpt_file_path: 储存训练状态的文件路径
    """
    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    tf.train.Saver().save(sess, ckpt_file_path+'.ckpt')


def load_checkpoint(ckpt_file_path):
    """恢复模型训练状态
    默认TensorFlow Session将从ckpt_file_path.ckpt中恢复所保存的训练状态
    Args:
        ckpt_file_path: 储存训练状态的文件路径
    """
    ckpt  = ckpt_file_path + '.ckpt'
    index = ckpt + ".index"
    meta  = ckpt + ".meta"
    if os.path.isfile(index) and os.path.isfile(meta):
        tf.train.Saver().restore(sess, ckpt)


def save_embedding(dictionary, network, embedding_file_path):
    """保存词向量
    将训练好的词向量保存到embedding_file_path.npy文件中
    Args:
        dictionary: 单词与单词ID映射表
            {'UNK': 0, '你': 1, '我': 2, ..., '别生气': 2545, '小姐姐': 2546, ...}
        network: 默认TensorFlow Session所初始化的网络结构
            network = tl.layers.InputLayer(x, name='input_layer')
            ...
        embedding_file_path: 储存词向量的文件路径
    Returns:
        单词与向量映射表以npy格式保存在embedding_file_path.npy文件中
        {'关注': [-0.91619176, -0.83772564, ..., -1.90845013,  0.74918884], ...}
    """
    words, ids = zip(*dictionary.items())
    params = network.normalized_embeddings
    embeddings = tf.nn.embedding_lookup(params, tf.constant(ids, dtype=tf.int32)).eval()
    wv = dict(zip(words, embeddings))
    path = os.path.dirname(os.path.abspath(embedding_file_path))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    tl.files.save_any_to_npy(save_dict=wv, name=embedding_file_path+'.npy')


def train(model_name):
    """训练词向量
    Args:
        corpus_file: 文件内容已经经过分词。
            得 我 就 在 车里 咪 一会
            终于 知道 哪里 感觉 不 对 了
            ...
        model_name: 模型名称，用于生成保存训练状态和词向量的文件名
    Returns:
        输出训练状态以及训练后的词向量文件
    """
    words           = load_dataset()
    data_size       = len(words)
    vocabulary_size = get_vocabulary_size(words, min_freq=3)
    batch_size      = 500  # 一次Forword运算以及BP运算中所需要的训练样本数目
    embedding_size  = 200  # 词向量维度
    skip_window     = 5    # 上下文窗口，单词前后各取五个词
    num_skips       = 10   # 从窗口中选取多少个预测对
    num_sampled     = 64   # 负采样个数
    learning_rate   = 0.1  # 学习率
    n_epoch         = 50   # 所有样本重复训练50次
    num_steps       = int((data_size/batch_size) * n_epoch) # 总迭代次数

    data, count, dictionary, reverse_dictionary = \
        tl.nlp.build_words_dataset(words, vocabulary_size)
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    with tf.device('/cpu:0'):
        emb_net = tl.layers.Word2vecEmbeddingInputlayer(
            inputs          = train_inputs,
            train_labels    = train_labels,
            vocabulary_size = vocabulary_size,
            embedding_size  = embedding_size,
            num_sampled     = num_sampled)
        loss = emb_net.nce_cost
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    sess.run(tf.global_variables_initializer())
    ckpt_file_path = "checkpoint/" + model_name
    load_checkpoint(ckpt_file_path)

    step = data_index = 0
    loss_vals = []
    while (step < num_steps):
        batch_inputs, batch_labels, data_index = tl.nlp.generate_skip_gram_batch(
            data=data, batch_size=batch_size, num_skips=num_skips,
            skip_window=skip_window, data_index=data_index)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
        loss_vals.append(loss_val)
        if (step != 0) and (step % 200) == 0:
            logging.info("(%d/%d) latest average loss: %f.", step, num_steps, sum(loss_vals)/len(loss_vals))
            del loss_vals[:]
            save_checkpoint(ckpt_file_path)
            embedding_file_path = "output/" + model_name
            save_embedding(dictionary, emb_net, embedding_file_path)
        step += 1


if __name__ == '__main__':
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    sess = tf.InteractiveSession() # 默认TensorFlow Session
    train('model_word2vec_200')
    sess.close()
