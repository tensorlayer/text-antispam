#! /usr/bin/env python3
"""这个例子用Dynamic Recurrent Neural Network (LSTM) 实现不定长文本序列分类。

Recurrent Neural Network for Text Classification with Multi-Task Learning: https://arxiv.org/pdf/1605.05101.pdf
A C-LSTM Neural Network for Text Classification: https://arxiv.org/pdf/1511.08630.pdf

"""

import logging
import math
import os
import random
import sys
import shutil
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from sklearn.model_selection import train_test_split


def load_dataset(files, test_size=0.2):
    """加载样本并取test_size的比例做测试集
    Args:
        files: 样本文件目录集合
            样本文件是包含了样本特征向量与标签的npy文件
        test_size: float
            0.0到1.0之间，代表数据集中有多少比例抽做测试集
    Returns:
        X_train, y_train: 训练集特征列表和标签列表
        X_test, y_test: 测试集特征列表和标签列表
    """
    x = []
    y = []
    for file in files:
        data = np.load(file)
        if x == [] or y == []:
            x = data['x']
            y = data['y']
        else:
            x = np.append(x, data['x'], axis=0)
            y = np.append(y, data['y'], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, y_train, x_test, y_test


def network(x, keep=0.8):
    """定义网络结构
    为了防止过拟合，我们对DynamicRNNLayer的输入与输出都做了Dropout操作。参数keep决定了输入或输出的保留比例。通过配置keep参数我们可以在训练的时候打开Dropout，在服务的时候关闭它。TensorFlow的tf.nn.dropout操作会根据keep值自动调整激活的神经元的输出权重，使得我们无需在keep改变时手动调节输出权重。
    Args:
        x: Input Placeholder
        keep: DynamicRNNLayer输入与输出神经元激活比例
            keep=1.0: 关闭Dropout
    Returns:
        network: 定义好的网络结构
    """
    n_hidden = 64 # hidden layer num of features
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DynamicRNNLayer(network,
        cell_fn         = tf.contrib.rnn.BasicLSTMCell,
        n_hidden        = n_hidden,
        dropout         = keep,
        sequence_length = tl.layers.retrieve_seq_length_op(x),
        return_seq_2d   = True,
        return_last     = True,
        name            = 'dynamic_rnn')
    network = tl.layers.DenseLayer(network, n_units=2,
                                   act=tf.identity, name="output")
    network.outputs_op = tf.argmax(tf.nn.softmax(network.outputs), 1)
    return network


def load_checkpoint(sess, ckpt_file):
    """恢复模型训练状态
    必须在tf.global_variables_initializer()之后
    """
    index = ckpt_file + ".index"
    meta  = ckpt_file + ".meta"
    if os.path.isfile(index) and os.path.isfile(meta):
        tf.train.Saver().restore(sess, ckpt_file)


def save_checkpoint(sess, ckpt_file):
    """保存模型训练状态
    """
    path = os.path.dirname(os.path.abspath(ckpt_file))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    tf.train.Saver().save(sess, ckpt_file)


def train(sess, x, network):
    """训练网络
    Args:
        sess: TensorFlow Session
        x: Input placeholder
        network: Network
    """
    learning_rate  = 0.1
    n_classes      = 1 # linear sequence or not
    y         = tf.placeholder(tf.int64, [None, ], name="labels")
    cost      = tl.cost.cross_entropy(network.outputs, y, 'xentropy')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct   = tf.equal(network.outputs_op, y)
    accuracy  = tf.reduce_mean(tf.cast(correct, tf.float32))

    # 使用TensorBoard可视化loss与准确率：`tensorboard --logdir=./logs`
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    writter_train = tf.summary.FileWriter('./logs/train', sess.graph)
    writter_test  = tf.summary.FileWriter('./logs/test')

    x_train, y_train, x_test, y_test = load_dataset(
        ["../word2vec/output/sample_seq_pass.npz",
         "../word2vec/output/sample_seq_spam.npz"])

    # initialize 必须在所有network和operation都定义完成之后
    # 在network定义之前initialize会报错：
    # FailedPreconditionError (see above for traceback): Attempting to use uninitialized value relu1/W
    #      [[Node: relu1/W/read = Identity[T=DT_FLOAT, _class=["loc:@relu1/W"], _device="/job:localhost/replica:0/task:0/gpu:0"](relu1/W)]]
    # 在optimizer定义之前initialize会报错：
    # FailedPreconditionError (see above for traceback): Attempting to use uninitialized value beta1_power
    #      [[Node: beta1_power/read = Identity[T=DT_FLOAT, _class=["loc:@relu1/W"], _device="/job:localhost/replica:0/task:0/gpu:0"](beta1_power)]]
    sess.run(tf.global_variables_initializer())
    load_checkpoint(sess, ckpt_file)

    n_epoch      = 2
    batch_size   = 128
    test_size    = 1280
    display_step = 10
    step         = 0
    total_step   = math.ceil(len(x_train) / batch_size) * n_epoch
    logging.info("batch_size: %d", batch_size)
    logging.info("Start training the network...")
    for epoch in range(n_epoch):
        for batch_x, batch_y in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=True):
            start_time = time.time()
            max_seq_len = max([len(d) for d in batch_x])
            for i,d in enumerate(batch_x):
                batch_x[i] += [np.zeros(200) for i in range(max_seq_len - len(d))]
            batch_x = list(batch_x) # ValueError: setting an array element with a sequence.

            feed_dict = {x: batch_x, y: batch_y}
            sess.run(optimizer, feed_dict)

            # TensorBoard打点
            summary = sess.run(merged, feed_dict)
            writter_train.add_summary(summary, step)

            # 计算测试集准确率
            start = random.randint(0, len(x_test)-test_size)
            test_data  = x_test[start:(start+test_size)]
            test_label = y_test[start:(start+test_size)]
            max_seq_len = max([len(d) for d in test_data])
            for i,d in enumerate(test_data):
                test_data[i] += [np.zeros(200) for i in range(max_seq_len - len(d))]
            test_data = list(test_data)
            summary = sess.run(merged, {x: test_data, y: test_label})
            writter_test.add_summary(summary, step)

            # 每十步输出loss值与准确率
            if step == 0 or (step + 1) % display_step == 0:
                logging.info("Epoch %d/%d Step %d/%d took %fs", epoch + 1, n_epoch,
                             step + 1, total_step, time.time() - start_time)
                loss = sess.run(cost, feed_dict=feed_dict)
                acc  = sess.run(accuracy, feed_dict=feed_dict)
                logging.info("Minibatch Loss= " + "{:.6f}".format(loss) +
                             ", Training Accuracy= " + "{:.5f}".format(acc))
                save_checkpoint(sess, ckpt_file)

            step += 1


def export(model_version, model_dir, sess, x, y_op):
    """导出tensorflow_serving可用的模型
    SavedModel（tensorflow.python.saved_model）提供了一种跨语言格式来保存和恢复训练后的TensorFlow模型。它使用方法签名来定义Graph的输入和输出，使上层系统能够更方便地生成、调用或转换TensorFlow模型。
    SavedModelBuilder类提供保存Graphs、Variables及Assets的方法。所保存的Graphs必须标注用途标签。在这个实例中我们打算将模型用于服务而非训练，因此我们用SavedModel预定义好的tag_constant.Serving标签。
    为了方便地构建签名，SavedModel提供了signature_def_utils API。我们通过signature_def_utils.build_signature_def()来构建predict_signature。一个predict_signature至少包含以下参数：
    * inputs  = {'x': tensor_info_x} 指定输入的tensor信息
    * outputs = {'y': tensor_info_y} 指定输出的tensor信息
    * method_name = signature_constants.PREDICT_METHOD_NAME
    method_name定义方法名，它的值应该是tensorflow/serving/predict、tensorflow/serving/classify和tensorflow/serving/regress三者之一。Builder标签用来明确Meta Graph被加载的方式，只接受serve和train两种类型。
    """
    if model_version <= 0:
        logging.warning('Please specify a positive value for version number.')
        sys.exit()

    path = os.path.dirname(os.path.abspath(model_dir))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)

    export_path = os.path.join(
        compat.as_bytes(model_dir),
        compat.as_bytes(str(model_version)))

    if os.path.isdir(export_path) == True:
        logging.warning('Path (%s) exists, removing directories...', export_path)
        shutil.rmtree(export_path)

    builder = saved_model_builder.SavedModelBuilder(export_path)
    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_y = utils.build_tensor_info(y_op)

    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'x': tensor_info_x},
        outputs={'y': tensor_info_y},
        # signature_constants.CLASSIFY_METHOD_NAME = "tensorflow/serving/classify"
        # signature_constants.PREDICT_METHOD_NAME  = "tensorflow/serving/predict"
        # signature_constants.REGRESS_METHOD_NAME  = "tensorflow/serving/regress"
        # 如果缺失method_name会报错：
        # grpc.framework.interfaces.face.face.AbortionError: AbortionError(code=StatusCode.INTERNAL, details="Expected prediction signature method_name to be one of {tensorflow/serving/predict, tensorflow/serving/classify, tensorflow/serving/regress}. Was: ")
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder.add_meta_graph_and_variables(
        sess,
        # tag_constants.SERVING  = "serve"
        # tag_constants.TRAINING = "train"
        # 如果只有train标签，TensorFlow Serving加载时会报错：
        # E tensorflow_serving/core/aspired_versions_manager.cc:351] Servable {name: default version: 2} cannot be loaded: Not found: Could not find meta graph def matching supplied tags.
        [tag_constants.SERVING],
        signature_def_map={
            'predict_text': prediction_signature,
            # 如果缺失会报错：
            # grpc.framework.interfaces.face.face.AbortionError: AbortionError(code=StatusCode.FAILED_PRECONDITION, details="Default serving signature key not found.")
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        })

    builder.save()


if __name__ == '__main__':
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    ckpt_file = "./rnn_checkpoint/rnn.ckpt"
    x = tf.placeholder("float", [None, None, 200], name="inputs")
    sess = tf.InteractiveSession()

    flags = tf.flags
    flags.DEFINE_string("mode", "train", "train or export")
    FLAGS = flags.FLAGS

    if FLAGS.mode == "train":
        network = network(x)
        train(sess, x, network)
        logging.info("Optimization Finished!")
    elif FLAGS.mode == "export":
        model_version = 1
        model_dir    = "./output/rnn_model"
        network = network(x, keep=1.0)
        sess.run(tf.global_variables_initializer())
        load_checkpoint(sess, ckpt_file)
        export(model_version, model_dir, sess, x, network.outputs_op)
        logging.info("Servable Export Finishied!")
