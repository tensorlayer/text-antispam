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
