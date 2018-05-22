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

### 其他常用方法

前文提到过，分类器还可以用NBOW+MLP（如图9所示）和CNN来实现。借助TensorLayer，我们可以很方便地重组网络。下面简单介绍这两种网络的结构及其实现。

由于词向量之间存在着线性平移的关系，如果相似词空间距离相近，那么在仅仅将文本中一个或几个词改成近义词的情况下，两个文本的词向量线性相加的结果也应该是非常接近的。

<div align="center">
<img src="../images/9-xxx-color.png">
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
<img src="../images/10-xxx-color.png">
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
