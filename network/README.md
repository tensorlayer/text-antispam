### 训练RNN分类器以及导出Servable

```
python3 rnn_classifier.py
python3 rnn_classifier.py --mode=export
```

执行`tensorboard --logdir=logs`可以观察loss与accuracy曲线

### 训练MLP分类器以及导出Servable

```
python3 mlp_classifier.py
python3 mlp_classifier.py --mode=export
```

### 训练CNN分类器以及导出Servable

```
python3 cnn_classifier.py
python3 cnn_classifier.py --mode=export
```

#### 训练分类器

我们使用Dynamic RNN实现不定长文本序列分类。首先加载数据，通过`sklearn`库的`train_test_split`方法将样本按照要求的比例切分成训练集和测试集。

```
import logging, math, os, random, sys, shutil, time
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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    return x_train, y_train, x_test, y_test
```

为了防止过拟合，我们对DynamicRNNLayer的输入与输出都做了Dropout操作。参数`keep`决定了输入或输出的保留比例。通过配置`keep`参数我们可以在训练的时候打开Dropout，在服务的时候关闭它。TensorFlow的`tf.nn.dropout`操作会根据`keep`值自动调整激活的神经元的输出权重，使得我们无须在`keep`改变时手动调节输出权重。

```
def network(x, keep=0.8):
    """定义网络结构
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
```

同样我们需要定时保存训练状态。

```
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
        logging.warning('(%s) not exists, making directories...', path)
        os.makedirs(path)
    tf.train.Saver().save(sess, ckpt_file)
```

例子中每一次迭代，我们给网络输入128条文本序列。根据预测结果与标签的差异，网络不断优化权重，减小损失，逐步提高分类的准确性。

```
def train(sess, x, network):
    """训练网络
    Args:
        sess: TensorFlow Session
        x: Input placeholder
        network: Network
    """
    learning_rate  = 0.1
    n_classes      = 1
    y         = tf.placeholder(tf.int64, [None, ], name="labels")
    cost      = tl.cost.cross_entropy(network.outputs, y, 'xentropy')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                        .minimize(cost)
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
        for batch_x, batch_y in tl.iterate.minibatches(
                x_train, y_train, batch_size, shuffle=True):
            start_time = time.time()
            max_seq_len = max([len(d) for d in batch_x])
            for i,d in enumerate(batch_x):
                batch_x[i] += \
                    [np.zeros(200) for i in range(max_seq_len - len(d))]
            batch_x = list(batch_x)

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
                test_data[i] += \
                    [np.zeros(200) for i in range(max_seq_len - len(d))]
            test_data = list(test_data)
            summary = sess.run(merged, {x: test_data, y: test_label})
            writter_test.add_summary(summary, step)

            # 每十步输出loss值与准确率
            if step == 0 or (step + 1) % display_step == 0:
                logging.info("Epoch %d/%d Step %d/%d took %fs",
                             epoch + 1, n_epoch, step + 1, total_step,
                             time.time() - start_time)
                loss = sess.run(cost, feed_dict=feed_dict)
                acc  = sess.run(accuracy, feed_dict=feed_dict)
#                logging.info("Minibatch Loss= " + "{:.6f}".format(loss) +
                             ", Training Accuracy= " + "{:.5f}".format(acc))
                save_checkpoint(sess, ckpt_file)

            step += 1
```

我们在训练过程中使用TensorBoard将Loss和Accuracy的变化可视化。如图5所示，在100步后，训练集与测试集的准确率都从最开始的50%左右上升到了95%以上。

<div align="center">
<img src="../images/5-Loss_and_Accuracy-color.png">
<br>
<em align="center">图5 使用TensorBoard监控Loss和Accuracy</em>
</div>

#### 模型导出

TensorFlow的SavedModel模块`tensorflow.python.saved_model`提供了一种跨语言格式来保存和恢复训练后的TensorFlow模型。它使用方法签名来定义Graph的输入和输出，使上层系统能够更方便地生成、调用或转换TensorFlow模型。SavedModelBuilder类提供保存Graphs、Variables及Assets的方法。所保存的Graphs必须标注用途标签。在这个实例中我们打算将模型用于服务而非训练，因此我们用SavedModel预定义好`tag_constant.Serving`标签。

为了方便构建签名，SavedModel提供了`signature_def_utils` API。我们通过`signature_def_utils.build_signature_def`方法来构建`predict_signature`。一个`predict_signature`至少包含以下参数：

```
inputs = {'x': tensor_info_x} 指定输入的tensor信息
outputs = {'y': tensor_info_y} 指定输出的tensor信息
method_name = signature_constants.PREDICT_METHOD_NAME
```

`method_name`定义方法名，它的值应该是`tensorflow/serving/predict`、`ten
sorflow/serving/classify`和`tensorflow/serving/regress`三者之一。Builder标签用来明确Meta Graph被加载的方式，只接受`serve`和`train`两种类型。接下来我们就要使用TensorFlow的SavedModelBuilder类来导出模型了。

```
def export(model_version, model_dir, sess, x, y_op):
    """导出tensorflow_serving可用的模型
    """
    if model_version <= 0:
        logging.warning('Please specify a positive value for version.')
        sys.exit()

    path = os.path.dirname(os.path.abspath(model_dir))
    if os.path.isdir(path) == False:
        logging.warning('(%s) not exists, making directories...', path)
        os.makedirs(path)

    export_path = os.path.join(
        compat.as_bytes(model_dir),
        compat.as_bytes(str(model_version)))

    if os.path.isdir(export_path) == True:
        logging.warning('(%s) exists, removing dirs...', export_path)
        shutil.rmtree(export_path)

    builder = saved_model_builder.SavedModelBuilder(export_path)
    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_y = utils.build_tensor_info(y_op)

    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'x': tensor_info_x},
        outputs={'y': tensor_info_y},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder.add_meta_graph_and_variables(
        sess,
        [tag_constants.SERVING],
        signature_def_map={
            'predict_text': prediction_signature,
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: \
                prediction_signature
        })

    builder.save()
```

以上函数准备完成，依次执行训练和导出，得到分类器服务模型（Servable）。


```
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
        model_dir = "./output/rnn_model"
        network = network(x, keep=1.0)
        sess.run(tf.global_variables_initializer())
        load_checkpoint(sess, ckpt_file)
        export(model_version, model_dir, sess, x, network.outputs_op)
        logging.info("Servable Export Finishied!")
```

我们将在`./output/rnn_model`目录下看到导出模型的每个版本，实例中`model_version`被设置为1，因此创建了相应的子目录`./output/rnn_mode/1`。

SavedModel目录具有以下结构。

```
assets/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb
```

导出的模型在TensorFlow Serving中又被称为Servable，其中`saved_model.pb`保存了接口的数据交换格式，`variables`保存了模型的网络结构和参数，`assets`用来保存如词库等模型初始化所需的外部资源。本例没有用到外部资源，因此没有`assets`文件夹。

### 其他常用方法

前文提到过，分类器还可以用NBOW+MLP（如图9所示）和CNN来实现。借助TensorLayer，我们可以很方便地重组网络。下面简单介绍这两种网络的结构及其实现。

由于词向量之间存在着线性平移的关系，如果相似词空间距离相近，那么在仅仅将文本中一个或几个词改成近义词的情况下，两个文本的词向量线性相加的结果也应该是非常接近的。

<div align="center">
<img src="../images/9-NBOW_and_MLP_Classifier-color.png">
<br>
<em align="center">图9 NBOW+MLP分类器</em>
</div>

多层神经网络可以无限逼近任意函数，能够胜任复杂的非线性分类任务。下面的代码将Word2vec训练好的词向量线性相加，再通过三层全连接网络进行分类。

```
def network(x, keep=0.8):
    """定义网络结构
    Args:
        x: Input Placeholder
        keep: 各层神经元激活比例
            keep=1.0: 关闭Dropout
    Returns:
        network: 定义好的网络结构
    """
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.DropoutLayer(
        network, keep=keep, name='drop1', is_fix=True)
    network = tl.layers.DenseLayer(
        network, n_units=200, act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(
        network, keep=keep, name='drop2', is_fix=True)
    network = tl.layers.DenseLayer(
        network, n_units=200, act=tf.nn.relu, name='relu')
    network = tl.layers.DropoutLayer(
        network, keep=keep, name='drop3', is_fix=True)
    network = tl.layers.DenseLayer(
        network, n_units=2, act=tf.identity, name='output')
    network.outputs_op = tf.argmax(tf.nn.softmax(network.outputs), 1)
    return network
```

CNN卷积的过程捕捉了文本的局部相关性，在文本分类中也取得了不错的结果。图10演示了CNN分类过程。输入是一个由6维词向量组成的最大长度为11的文本，经过与4个3×6的卷积核进行卷积，得到4张9维的特征图。再对特征图每3块不重合区域进行最大池化，将结果合成一个12维的向量输入到全连接层。

<div align="center">
<img src="../images/10-CNN_Classifier-color.png">
<br>
<em align="center">图10 CNN分类器</em>
</div>

下面代码中输入是一个由200维词向量组成的最大长度为20的文本（确定好文本的最大长度后，我们需要对输入进行截取或者填充）。卷积层参数[3, 200, 6]代表6个3×200的卷积核。这里使用1D CNN，是因为我们把文本序列看成一维数据，这意味着卷积的过程只会朝一个方向进行（同理，处理图片和小视频分别需要使用2D CNN和3D CNN）。卷积核宽度被设置为和词向量大小一致，确保了词向量作为最基本的元素不会被破坏。我们选取连续的3维作为池化区域，滑动步长取3，使得池化区域不重合，最后通过一个带Dropout的全连接层得到Softmax后的输出。

```
def network(x, keep=0.8):
    """定义网络结构
    Args:
        x: Input Placeholder
        keep: 全连接层输入神经元激活比例
            keep=1.0: 关闭Dropout
    Returns:
        network: 定义好的网络结构
    """
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv1dLayer(
        network, act=tf.nn.relu, shape=[3, 200, 6],
        name='cnn_layer1', padding='VALID')
    network = tl.layers.MaxPool1d(
        network, filter_size=3, strides=3, name='pool_layer1')
    network = tl.layers.FlattenLayer(
        network, name='flatten_layer')
    network = tl.layers.DropoutLayer(
        network, keep=keep, name='drop1', is_fix=True)
    network = tl.layers.DenseLayer(
        network, n_units=2, act=tf.identity, name="output")
    network.outputs_op = tf.argmax(tf.nn.softmax(network.outputs), 1)
    return network
```
