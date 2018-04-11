
### TensorFlow Serving部署

反垃圾服务分为线上与线下两层。线上实时服务要求毫秒级判断文本是否属于垃圾文本，线下离线计算需要根据新进的样本不断更新模型，并及时推送到线上。

图6所示的分类器就是用TensorFlow Serving提供的服务。TensorFlow Serving是一个灵活、高性能的机器学习模型服务系统，专为生产环境而设计。它可以将训练好的机器学习模型轻松部署到线上，并且支持热更新。它使用gRPC作为接口框架接受外部调用，服务稳定，接口简单。这些优秀特性使我们能够专注于线下模型训练。

<div align="center">
<img src="../images/6-Antispam_Service_Architecture-color.png">
<br>
<em align="center">图6 反垃圾服务架构</em>
</div>

为什么使用TensorFlow Serving而不是直接启动多个加载了模型的Python进程来提供线上服务？因为重复引入TensorFlow并加载模型的Python进程浪费资源并且运行效率不高。而且TensorFlow本身有一些限制导致并不是所有时候都能启动多个进程。TensorFlow默认会使用尽可能多的GPU并且占用所使用的GPU。因此如果有一个TensorFlow进程正在运行，可能导致其他TensorFlow进程无法启动。虽然可以指定程序使用特定的GPU，但是进程的数量也受到GPU数量的限制，总体来说不利于分布式部署。而TensorFlow Serving提供了一个高效的分布式解决方案。当新数据可用或改进模型时，加载并迭代模型是很常见的。TensorFlow Serving能够实现模型生命周期管理，它能自动检测并加载最新模型或回退到上一个模型，非常适用于高频迭代场景。

TensorFlow Serving的编译依赖Google的开源编译工具Bazel。具体的安装可以参考[官方文档](https://docs.bazel.build/versions/master/install-compile-source.html)。

部署的方式非常简单，只需在启动TensorFlow Serving时加载Servable并定义`model_name`即可，这里的`model_name`将用于与客户端进行交互。

```
$ ./tensorflow_model_server --port=9000 --model_base_path=./model --
model_name=antispam
```

可以看到TensorFlow Serving成功加载了我们刚刚导出的模型。

```
I tensorflow_serving/model_servers/server_core.cc:338] Adding/
updating models.
I tensorflow_serving/model_servers/server_core.cc:384] (Re-)adding 
model:antispam
I .../basic_manager.cc:698] Successfully reserved resources to load
servable {name: antispam version: 1}
...
I external/.../saved_model/loader.cc:274] Loading SavedModel: 
success.
Took 138439 microseconds.
I .../loader_harness.cc:86] Successfully loaded servable version
{name: antispam version: 1}
I tensorflow_serving/model_servers/main.cc:298] Running ModelServer
at 0.0.0.0:9000 ...
```

### 客户端调用

TensorFlow Serving通过gRPC框架接收外部调用。gRPC是一种高性能、通用的远程过程调用（Remote Procedure Call，RPC）框架。RPC协议包含了编码协议和传输协议。gRPC的编码协议是Protocol Buffers（ProtoBuf），它是Google开发的一种二进制格式数据描述语言，支持众多开发语言和平台。与JSON、XML相比，ProtoBuf的优点是体积小、速度快，其序列化与反序列化代码都是通过代码生成器根据定义好的数据结构生成的，使用起来也很简单。gRPC的传输协议是HTTP/2，相比于HTTP/1.1，HTTP/2引入了头部压缩算法（HPACK）等新特性，并采用了二进制而非明文来打包、传输客户端——服务器间的数据，性能更好，功能更强。总而言之，gRPC提供了一种简单的方法来精确地定义服务，并自动为客户端生成可靠性很强的功能库，如图7所示。

在使用gRPC进行通信之前，我们需要完成两步操作：1）定义服务；2）生成服务端和客户端代码。定义服务这块工作TensorFlow Serving已经帮我们完成了。在[TensorFlow Serving](https://github.com/tensorflow/serving)项目中，我们可以在以下目录找到三个`.proto`文件：`model.proto`、`predict.proto`和`prediction_service.proto`。这三个`.proto`文件定义了一次预测请求的输入和输出。例如一次预测请求应该包含哪些元数据（如模型的名称和版本），以及输入、输出与Tensor如何转换。

<div align="center">
<img src="../images/7-gRPC-color.png">
<br>
<em align="center">图7 客户端与服务端使用gRPC进行通信</em>
</div>

```
$ tree serving
serving
├── tensorflow
│   ├── ...
├── tensorflow_serving
│   ├── apis
│   │   ├── model.proto
│   │   ├── predict.proto
│   │   ├── prediction_service.proto
│   │   ├── ...
│   ├── ...
├── ...
```


接下来需要生成Python可以直接调用的功能库。首先将这三个文件复制到`serving/
tensorflow`目录下:

```
$ cd serving/tensorflow_serving/apis
$ cp model.proto predict.proto prediction_service.proto ../../tensorflow
```


因为我们移动了文件，所以`predict.proto`和`prediction_service.proto`的`import`需要略作修改：

```
predict.proto: import "tensorflow_serving/apis/model.proto" 
-> import "model.proto"
prediction_service.proto: import "tensorflow_serving/apis/predict.proto" 
-> import "predict.proto"
```


删去没有用到的RPC定义`service (Classify, Regress, GetModelMetadata)`和引入`import (classification.proto, get_model_metadata.proto, regression.proto)`。最后使用`grpcio-tools`生成功能库。

```
$ pip install grpcio
$ pip install grpcio-tools
$ python -m grpc.tools.protoc -I./ --python_out=. --grpc_python_out=. ./
*.proto
```


在当前目录能找到以下6个文件：

```
model_pb2.py
model_pb2_grpc.py
predict_pb2.py
predict_pb2_grpc.py
prediction_service_pb2.py
prediction_service_pb2_grpc.py
```

其中`model_pb2.py`、`predict_pb2.py`和`prediction_service_pb2.py`是Python与TensorFlow Serving交互所必需的功能库。

接下来写一个简单的客户端程序来调用部署好的模型。`engine.py`负责构建一个Request用于与TensorFlow Serving交互。为了描述简洁，这里分词使用了结巴分词，词向量也是直接载入内存，实际生产环境中分词与词向量获取是一个单独的服务。特别需要注意的是，输入的签名和数据必须与之前导出的模型相匹配，如图8所示。

<div align="center">
<img src="../images/8-From_Client_to_Server-color.png">
<br>
<em align="center">图8 从客户端到服务端</em>
</div>

```
import numpy as np
import jieba
import tensorlayer as tl
from grpc.beta import implementations
import predict_pb2
import prediction_service_pb2

def text_tensor(text, wv):
    """获取文本向量
    Args:
        text: 待检测文本
        wv: 词向量模型
    Returns:
        [[[ 3.80905056   1.94315064  -0.20703495  -1.31589055   1.9627794
           ...
           2.16935492   2.95426321  -4.71534014  -3.25034237 -11.28901672]]]
    """
    words = jieba.cut(text.strip())
    text_sequence = []
    for word in words:
        try:
            text_sequence.append(wv[word])
        except KeyError:
            text_sequence.append(wv['UNK'])
    text_sequence = np.asarray(text_sequence)
    sample = text_sequence.reshape(1, len(text_sequence), 200)
    return sample

print(" ".join(jieba.cut('分词初始化')))
wv = tl.files.load_npy_to_any(
    name='../word2vec/output/model_word2vec_200.npy')

host, port = ('localhost', '9000')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'antispam'
```

`serving.py`负责接收和处理请求。生产环境中一般使用反向代理软件如Nginx实现负载均衡。这里我们演示直接监听80端口来提供HTTP服务。


```
import json
import tornado.ioloop
import tornado.web
import tensorflow as tf
import engine

class MainHandler(tornado.web.RequestHandler):
    """请求处理类
    """

    def get(self):
        """处理GET请求
        """
        text = self.get_argument("text")
        predict = self.classify(text)
        data = {
            'text' : text,
            'predict' : predict[0]
        }
        self.write(json.dumps({'data': data}))

    def classify(self, text):
        """调用引擎检测文本
        Args:
            text: 待检测文本
        Returns:
            垃圾返回[0]，通过返回[1]
        """
        sample = engine.text_tensor(text, engine.wv)
        tensor_proto = tf.contrib.util.make_tensor_proto(
            sample, shape=[1, len(sample[0]), 200])
        engine.request.inputs['x'].CopyFrom(tensor_proto)
        response = engine.stub.Predict(engine.request, 10.0) # 10s timeout
        result = list(response.outputs['y'].int64_val)
        return result

def make_app():
    """定义并返回Tornado Web Application
    """
    return tornado.web.Application([
        (r"/predict", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(80)
    print("listen start")
    tornado.ioloop.IOLoop.current().start()
```

如果是在本地启动服务，访问`http://127.0.0.1:8021/predict?text=加我微信xxxxx有福利`，可以看到如下结果。

```
{
    "data": {
        "text": "加我微信xxxxx有福利",
        "predict": 0
    }
}
```

成功识别出垃圾消息。

### Centos7编译TensorFlow Serving

如果遇到“tar (child): bzip2：无法 exec: 没有那个文件或目录”错误：

```
yum install bzip2
```

如果遇到“C++ compilation of rule '@curl//:curl' failed”错误：

```
touch /usr/include/stropts.h
```

如果启动serving.py时遇到“ModuleNotFoundError: No module named 'PyQt4'”：

```
yum install pyqt4
```
