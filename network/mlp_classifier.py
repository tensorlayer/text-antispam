#! /usr/bin/env python3
"""训练Neural Bag-of-Words (NBOW) + Multilayer Perceptron (MLP)分类器

Bag of Tricks for Efficient Text Classification: https://arxiv.org/pdf/1607.01759.pdf
Neural Bag-of-Ngrams: http://iir.ruc.edu.cn/~libf/papers/AAAI-17.pdf

"""
import logging
import math
import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from sklearn.model_selection import train_test_split
import h5py
import json
from tensorflow.python.util import serialization
import tensorflow.keras as keras


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


def get_model(input_shape, keep=0.5):
    """定义网络结构
    为了防止过拟合，我们在读取前一层输出之前都加了Dropout操作。参数keep决定了某一层神经元输出保留的比例。通过配置keep我们可以在训练的时候打开Dropout，在测试的时候关闭它。TensorFlow的tf.nn.dropout操作会自动根据keep调整激活的神经元的输出权重，使得我们无需在keep改变时手动调节输出权重。
    Args:
        inputs_shape: 输入数据的shape
        keep: 各层神经元激活比例
            keep=1.0: 关闭Dropout
    Returns:
        model: 定义好的网络结构
    """
    ni = tl.layers.Input(input_shape, name='input_layer')
    nn = tl.layers.Dropout(keep=keep, name='drop1')(ni)
    nn = tl.layers.Dense(n_units=200, act=tf.nn.relu, name='relu1')(nn)
    nn = tl.layers.Dropout(keep=keep, name='drop2')(nn)
    nn = tl.layers.Dense(n_units=200, act=tf.nn.relu, name='relu2')(nn)
    nn = tl.layers.Dropout(keep=keep, name='drop3')(nn)
    nn = tl.layers.Dense(n_units=2, act=tf.nn.relu, name='output_layer')(nn)
    model = tl.models.Model(inputs=ni, outputs=nn, name='mlp')
    return model


def accuracy(y_pred, y_true):
    """
    计算预测精准度accuracy
    :param y_pred: 模型预测结果
    :param y_true: 真实结果 ground truth
    :return: 精准度acc
    """
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def train(model):
    # 训练网络
    learning_rate = 0.01
    n_epoch = 50
    batch_size = 128
    loss_vals = []
    acc_vals = []
    display_step = 10
    logging.info("batch_size: %d", batch_size)
    logging.info("Start training the network...")
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.train()

    for epoch in range(n_epoch):
        step = 0
        total_step = math.ceil(len(x_train) / batch_size)
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=True):
            X_train_a = tf.convert_to_tensor(X_train_a, dtype=tf.float32)

            with tf.GradientTape() as tape:
                _y = model(X_train_a)
                loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(y_train_a.astype(np.int32), _y, name='train_loss')
                loss_val = tf.reduce_mean(loss_val)
            grad = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))

            loss_vals.append(loss_val)
            acc_vals.append(accuracy(_y, y_train_a))
            if step + 1 == 1 or (step + 1) % display_step == 0:
                logging.info("Epoch {}/{},Step {}/{}, took {}".format(epoch + 1, n_epoch, step, total_step,
                                                                      time.time() - start_time))
                loss = sum(loss_vals) / len(loss_vals)
                acc = sum(acc_vals) / len(acc_vals)
                del loss_vals[:]
                del acc_vals[:]
                logging.info("Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1

        model.eval()
        test_loss, test_acc, n_iter = 0, 0, 0
        for batch_x, batch_y in tl.iterate.minibatches(x_test, y_test, batch_size, shuffle=True):
            batch_y = batch_y.astype(np.int32)
            _y = model(batch_x)

            loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(batch_y, _y, name='test_loss')
            loss_val = tf.reduce_mean(loss_val)

            test_loss += loss_val
            test_acc += accuracy(_y, batch_y)
            n_iter += 1
        logging.info("   test loss: {}".format(test_loss / n_iter))
        logging.info("   test acc:  {}".format(test_acc / n_iter))


def layer_conv1d_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    filters = args['n_filter']
    kernel_size = [args['filter_size']]
    strides = [args['stride']]
    padding = args['padding']
    data_format = args['data_format']
    dilation_rate = [args['dilation_rate']]
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'filters': filters,
              'kernel_size': kernel_size, 'strides': strides, 'padding': padding, 'data_format': data_format,
              'dilation_rate': dilation_rate, 'activation': 'relu', 'use_bias': True,
              'kernel_initializer': {'class_name': 'GlorotUniform',
                                     'config': {'seed': None}
                                     },
              'bias_initializer': {'class_name': 'Zeros',
                                   'config': {}
                                   },
              'kernel_regularizer': None, 'bias_regularizer': None,
              'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}

    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'Conv1D', 'config': config}
    return result


def layer_maxpooling1d_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    pool_size = [args['filter_size']]
    strides = [args['strides']]
    padding = args['padding']
    data_format = args['data_format']
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'strides': strides, 'pool_size': pool_size,
              'padding': padding, 'data_format': data_format}
    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'MaxPooling1D', 'config': config}
    return result


def layer_flatten_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']

    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'data_format': 'channels_last'}
    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'Flatten', 'config': config}
    return result


def layer_dropout_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    rate = 1-args['keep']
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'rate': rate, 'noise_shape': None, 'seed': None}
    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'Dropout', 'config': config}
    return result


def layer_dense_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    units = args['n_units']
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'units': units, 'activation': 'relu', 'use_bias': True,
              'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
              'bias_initializer': {'class_name': 'Zeros', 'config': {}},
              'kernel_regularizer': None,
              'bias_regularizer': None,
              'activity_regularizer': None,
              'kernel_constraint': None,
              'bias_constraint': None}

    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'Dense', 'config': config}
    return result


def layer_rnn_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    cell = {'class_name': 'LSTMCell', 'config': {'name': 'lstm_cell', 'trainable': True, 'dtype': 'float32',
                                                 'units': 64, 'activation': 'tanh', 'recurrent_activation': 'sigmoid',
                                                 'use_bias': True,
                                                 'kernel_initializer': {'class_name': 'GlorotUniform',
                                                                        'config': {'seed': None}},
                                                 'recurrent_initializer': {'class_name': 'Orthogonal',
                                                                           'config': {'gain': 1.0, 'seed': None}},
                                                 'bias_initializer': {'class_name': 'Zeros', 'config': {}},
                                                 'unit_forget_bias': True, 'kernel_regularizer': None,
                                                 'recurrent_regularizer': None, 'bias_regularizer': None,
                                                 'kernel_constraint': None, 'recurrent_constraint': None,
                                                 'bias_constraint': None, 'dropout': 0.0, 'recurrent_dropout': 0.2,
                                                 'implementation': 1}}
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'return_sequences': False,
              'return_state': False, 'go_backwards': False, 'stateful': False, 'unroll': False, 'time_major': False,
              'cell': cell
              }
    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'RNN', 'config': config}
    return result


def layer_translator(tl_layer, is_first_layer=False):
    _input_shape = None
    global input_shape
    if is_first_layer:
        _input_shape = input_shape
    if tl_layer['class'] == '_InputLayer':
        input_shape = tl_layer['args']['shape']
    elif tl_layer['class'] == 'Conv1d':
        return layer_conv1d_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'MaxPool1d':
        return layer_maxpooling1d_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'Flatten':
        return layer_flatten_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'Dropout':
        return layer_dropout_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'Dense':
        return layer_dense_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'RNN':
        return layer_rnn_translator(tl_layer, _input_shape)
    return None


def config_translator(f_tl, f_k):
    tl_model_config = f_tl.attrs['model_config'].decode('utf8')
    tl_model_config = eval(tl_model_config)
    tl_model_architecture = tl_model_config['model_architecture']

    k_layers = []
    for key, tl_layer in enumerate(tl_model_architecture):
        if key == 1:
            k_layer = layer_translator(tl_layer, is_first_layer=True)
        else:
            k_layer = layer_translator(tl_layer)
        if k_layer is not None:
            k_layers.append(k_layer)
    f_k.attrs['model_config'] = json.dumps({'class_name': 'Sequential',
                                            'config': {'name': 'sequential', 'layers': k_layers},
                                            'build_input_shape': input_shape},
                                           default=serialization.get_json_type).encode('utf8')
    f_k.attrs['backend'] = keras.backend.backend().encode('utf8')
    f_k.attrs['keras_version'] = str(keras.__version__).encode('utf8')

    # todo: translate the 'training_config'
    training_config = {'loss': {'class_name': 'SparseCategoricalCrossentropy',
                                'config': {'reduction': 'auto', 'name': 'sparse_categorical_crossentropy',
                                           'from_logits': False}},
                       'metrics': ['accuracy'], 'weighted_metrics': None, 'loss_weights': None, 'sample_weight_mode': None,
                       'optimizer_config': {'class_name': 'Adam',
                                            'config': {'name': 'Adam', 'learning_rate': 0.01, 'decay': 0.0,
                                                       'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False
                                                       }
                                            }
                       }

    f_k.attrs['training_config'] = json.dumps(training_config, default=serialization.get_json_type).encode('utf8')


def weights_translator(f_tl, f_k):
    # todo: delete inputlayer
    if 'model_weights' not in f_k.keys():
        f_k_model_weights = f_k.create_group('model_weights')
    else:
        f_k_model_weights = f_k['model_weights']
    for key in f_tl.keys():
        if key not in f_k_model_weights.keys():
            f_k_model_weights.create_group(key)
        try:
            f_tl_para = f_tl[key][key]
        except KeyError:
            pass
        else:
            if key not in f_k_model_weights[key].keys():
                f_k_model_weights[key].create_group(key)
            weight_names = []
            f_k_para = f_k_model_weights[key][key]
            # todo：对RNN层的weights进行通用适配
            cell_name = ''
            if key == 'rnn_1':
                cell_name = 'lstm_cell'
                f_k_para.create_group(cell_name)
                f_k_para = f_k_para[cell_name]
                f_k_model_weights.create_group('masking')
                f_k_model_weights['masking'].attrs['weight_names'] = []
            for k in f_tl_para:
                if k == 'biases:0' or k == 'bias:0':
                    weight_name = 'bias:0'
                elif k == 'filters:0' or k == 'weights:0' or k == 'kernel:0':
                    weight_name = 'kernel:0'
                elif k == 'recurrent_kernel:0':
                    weight_name = 'recurrent_kernel:0'
                else:
                    raise Exception("cant find the parameter '{}' in tensorlayer".format(k))
                if weight_name in f_k_para:
                    del f_k_para[weight_name]
                f_k_para.create_dataset(name=weight_name, data=f_tl_para[k][:],
                                                           shape=f_tl_para[k].shape)

        weight_names = []
        for weight_name in f_tl[key].attrs['weight_names']:
            weight_name = weight_name.decode('utf8')
            weight_name = weight_name.split('/')
            k = weight_name[-1]
            if k == 'biases:0' or k == 'bias:0':
                weight_name[-1] = 'bias:0'
            elif k == 'filters:0' or k == 'weights:0' or k == 'kernel:0':
                weight_name[-1] = 'kernel:0'
            elif k == 'recurrent_kernel:0':
                weight_name[-1] = 'recurrent_kernel:0'
            else:
                raise Exception("cant find the parameter '{}' in tensorlayer".format(k))
            if key == 'rnn_1':
                weight_name.insert(-1, 'lstm_cell')
            weight_name = '/'.join(weight_name)
            weight_names.append(weight_name.encode('utf8'))
        f_k_model_weights[key].attrs['weight_names'] = weight_names

    f_k_model_weights.attrs['backend'] = keras.backend.backend().encode('utf8')
    f_k_model_weights.attrs['keras_version'] = str(keras.__version__).encode('utf8')

    f_k_model_weights.attrs['layer_names'] = [i for i in f_tl.attrs['layer_names']]


def translator_tl2_keras_h5(_tl_h5_path, _keras_h5_path):
    f_tl_ = h5py.File(_tl_h5_path, 'r+')
    f_k_ = h5py.File(_keras_h5_path, 'a')
    f_k_.clear()
    weights_translator(f_tl_, f_k_)
    config_translator(f_tl_, f_k_)
    f_tl_.close()
    f_k_.close()


if __name__ == '__main__':
    # 定义log格式
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)

    # 加载数据
    x_train, y_train, x_test, y_test = load_dataset(
        ["../word2vec/output/sample_pass.npz",
        "../word2vec/output/sample_spam.npz"])

    # 构建模型
    input_shape = [None, 200]
    model = get_model(input_shape)

    # 开始训练
    train(model)
    logging.info("Optimization Finished!")

    # h5保存和转译
    model_dir = './model_h5'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    tl_h5_path = model_dir + '/model_mlp_tl.hdf5'
    keras_h5_path = model_dir + '/model_mlp_tl2k.hdf5'
    tl.files.save_hdf5_graph(network=model, filepath=tl_h5_path, save_weights=True)
    translator_tl2_keras_h5(tl_h5_path, keras_h5_path)

    # 读取模型
    new_model = keras.models.load_model(keras_h5_path)
    score = new_model.evaluate(x_test, y_test, batch_size=128)

    # 保存SavedModel可部署文件
    saved_model_version = 1
    saved_model_path = "./saved_models/mlp/"
    tf.saved_model.save(new_model, saved_model_path + str(saved_model_version))