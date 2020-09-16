#! /usr/bin/env python3
"""与加载了RNN Classifier导出的Servable的TensorFlow Serving进行通信
"""

import json
import tornado.ioloop
import tornado.web
import tensorflow as tf
import jieba
import tensorlayer as tl
from packages import text_regularization as tr
import numpy as np
import requests


print(" ".join(jieba.cut('分词初始化')))
wv = tl.files.load_npy_to_any(name='../word2vec/output/model_word2vec_200.npy')


def text_tensor(text, wv):
    """获取文本向量
    Args:
        text: 待检测文本
        wv: 词向量模型
    Returns:
        [[[ 3.80905056   2.94315064  -0.20703495  -2.31589055   2.9627794
           ...
           2.16935492   2.95426321  -4.71534014  -3.25034237 -11.28901672]]]
    """
    text = tr.extractWords(text)
    words = jieba.cut(text.strip())
    text_embedding = np.zeros(200)
    for word in words:
        try:
            text_embedding += wv[word]
        except KeyError:
            text_embedding += wv['UNK']
    text_embedding = np.asarray(text_embedding)
    sample = text_embedding.reshape(1, 200)
    return sample


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
            'predict' : str(predict[0])
        }
        self.write(json.dumps({'data': data}))

    def classify(self, text):
        """调用引擎检测文本
        Args:
            text: 待检测文本
        Returns:
            垃圾返回[0]，通过返回[1]
        """
        sample = text_tensor(text, wv)
        sample = np.array(sample, dtype=np.float32)

        sample = sample.reshape(1, 200)
        data = json.dumps({
            # "signature_name": "call",
            "instances": sample.tolist()
        })
        headers = {"content-type": "application/json"}
        json_response = requests.post(
            'http://localhost:8501/v1/models/saved_model:predict',
            data=data, headers=headers)

        predictions = np.array(json.loads(json_response.text)['predictions'])
        result = np.argmax(predictions, axis=-1)
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
