#! /usr/bin/env python3
"""与加载了CNN Classifier导出的Servable的TensorFlow Serving进行通信
"""

import json
import tornado.ioloop
import tornado.web
import tensorflow as tf

import engine_for_cnn as engine


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
        tensor_proto = tf.contrib.util.make_tensor_proto(sample, shape=[1, len(sample[0]), 200])
        engine.request.inputs['x'].CopyFrom(tensor_proto)
        response = engine.stub.Predict(engine.request, 10.0)  # 10 secs timeout
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
