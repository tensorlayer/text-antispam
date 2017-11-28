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

### 与TensorFlow Serving进行通信

TensorFlow Serving 通过gRPC 框架接收外部调用。gRPC 是一种高性能、通用的远程过程调用（Remote Procedure Call，RPC）框架。RPC 协议包含了编码协议和传输协议。gRPC 的编码协议是Protocol Buffers（ProtoBuf），它是Google 开发的一种二进制格式数据描述语言，支持众多开发语言和平台。与JSON、XML 相比，ProtoBuf 的优点是体积小、速度快，其序列化与反序列化代码都是通过代码生成器根据定义好的数据结构生成的，使用起来也很简单。gRPC 的传输协议是HTTP/2，相比于HTTP/1.1，HTTP/2引入了头部压缩算法（HPACK）等新特性，并采用了二进制而非明文来打包、传输客户端——服务器间的数据，性能更好，功能更强。总而言之，gRPC 提供了一种简单的方法来精确地定义服务，并自动为客户端生成可靠性很强的功能库。

在使用gRPC 进行通信之前，我们需要完成两步操作：1）定义服务；2）生成服务端和客户端代码。定义服务这块工作TensorFlow Serving 已经帮我们完成了。在[TensorFlow Serving](https://github.com/tensorflow/serving) 项目中，我们可以在以下目录找到
三个.proto 文件：model.proto、predict.proto 和prediction_service.proto。这三个.proto 文件定义了一次预测请求的输入和输出。例如一次预测请求应该包含哪些元数据（如模型的名称和版本），以及输入、输出与Tensor 如何转换。

```
$ tree serving
serving
├── tensorflow
│ ├── ...
├── tensorflow_serving
│ ├── apis
│ │ ├── model.proto
│ │ ├── predict.proto
│ │ ├── prediction_service.proto
│ │ ├── ...
│ ├── ...
├── ...
```

接下来需要生成Python 可以直接调用的功能库。首先将这三个文件复制到serving/tensorflow 目录下:

```
$ cd serving/tensorflow_serving/apis
$ cp model.proto predict.proto prediction_service.proto ../../tensorflow
```

因为我们移动了文件，所以predict.proto 和prediction_service.proto 的import 需要略作修改：

```
predict.proto: import "tensorflow_serving/apis/model.proto"
-> import "model.proto"
prediction_service.proto: import "tensorflow_serving/apis/predict.proto"
-> import "predict.proto"
```

删去没有用到的RPC 定义service (Classify, Regress, GetModelMetadata)和引入import (classification.proto, get_model_metadata.proto, regression.proto)。最后使用grpcio-tools 生成功能库。

```
$ pip install grpcio
$ pip install grpcio-tools
$ python -m grpc.tools.protoc -I./ --python_out=. --grpc_python_out=. ./
*.proto
```

在当前目录能找到以下6 个文件：

```
model_pb2.py
model_pb2_grpc.py
predict_pb2.py
predict_pb2_grpc.py
prediction_service_pb2.py
prediction_service_pb2_grpc.py
```

其中model_pb2.py、predict_pb2.py 和prediction_service_pb2.py 是Python 与TensorFlow Serving 交互所必需的功能库。

```python
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
```
