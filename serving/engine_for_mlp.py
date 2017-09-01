#! /usr/bin/env python3
"""与加载了NBOW+MLP Classifier导出的Servable的TensorFlow Serving进行通信
"""

import numpy as np
import jieba
import tensorlayer as tl
from grpc.beta import implementations

import predict_pb2
import prediction_service_pb2
from packages import text_regularization as tr


def text_tensor(text, wv):
    """获取文本向量
    Args:
        text: 待检测文本
        wv: 词向量模型
    Returns:
        [[ 3.80905056   1.94315064  -0.20703495  -1.31589055   1.9627794
           ...
           2.16935492   2.95426321  -4.71534014  -3.25034237 -11.28901672]]
    """
    text = tr.extractWords(text)
    words = jieba.cut(text.strip())
    text_embedding = np.zeros(200, dtype=np.float32)
    for word in words:
        try:
            text_embedding += wv[word]
        except KeyError:
            text_embedding += wv['UNK']

    sample = text_embedding.reshape(1, 200)
    return sample


print(" ".join(jieba.cut('分词初始化')))
wv = tl.files.load_npy_to_any(name='../word2vec/output/model_word2vec_200.npy')

host, port = ('localhost', '9000')
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'antispam'
