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

```
pip install grpcio
pip install grpcio-tools
```

生成predict pb2文件需要serving/tensorflow_serving/apis下三个proto，model.proto, predict.proto, prediction_service.proto

predict.proto和prediction_service.proto的import需要略作修改：

```
predict.proto: import "tensorflow_serving/apis/model.proto" -> import "model.proto"
prediction_service.proto:
    import "tensorflow_serving/apis/predict.proto" -> import "predict.proto"
    删去其他import(classification.proto, get_model_metadata.proto, regression.proto)
    删去对应的rpc定义(Classify, Regress, GetModelMetadata)
```

将model.proto, predict.proto, prediction_service.proto copy到serving/tensorflow目录下，执行

```
python -m grpc.tools.protoc -I./ --python_out=.. --grpc_python_out=.. ./*.proto
```

在上级目录能找到：

```
model_pb2.py
model_pb2_grpc.py
predict_pb2.py
predict_pb2_grpc.py
prediction_service_pb2.py
prediction_service_pb2_grpc.py
```

model_pb2.py, predict_pb2.py, prediction_service_pb2.py是python与tensorflow_serving交互所必需的：

```python
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
```
