
- [一个自然语言情感分类项目。](#一个自然语言情感分类项目)
- [项目的流程为：](#项目的流程为)
- [目录结构说明](#目录结构说明)
    - [project 里为项目神经网络训练和预测的代码](#project-里为项目神经网络训练和预测的代码)
    - [tools 里是一些辅助程序](#tools-里是一些辅助程序)
    - [weibo\_spider 里为python爬虫程序。](#weibo_spider-里为python爬虫程序)
    - [nn目录下的其他\*.py文件](#nn目录下的其他py文件)
- [说明](#说明)


## 一个自然语言情感分类项目。
## 项目的流程为：
1. 首先使用python爬虫爬取微博上的网民言论。
2. 将爬取完的句子进行清洗、标注情感标签 。
3. 使用预训练的BERT模型对句子进行编码，得到含有整个句子信息的特征向量序列，然后将特征向量序列放入LSTM网络进行训练。
4. 用训练完的LSTM和预训练的BERT模型来进行预测，将网民的言论按情感分类，最后的到情感分布。
5. 最后将结果通过web网页的方式进行可视化展示。
## 目录结构说明
#### project 里为项目神经网络训练和预测的代码
#### tools 里是一些辅助程序
Sniffer.py 递归地探测指定pattern的文件
Transformer.py 用于数据格式转化
activate* 是激活函数和以及激活函数的微分
decorator_set.py 为计时器修饰器
resource_monitor.py 用于资源监测
#### weibo_spider 里为python爬虫程序。
#### nn目录下的其他*.py文件
dnn.py,cnn.py,rnn.py,main.py,process_data.py 与项目无关。
其中dnn.py,rnn.py 是手动实现神经网络模型。
## 说明
项目采用pytoch框架，项目的训练和预测过程在debian-GPU服务器上进行。数据爬取和清洗过程在个人PC上完成。
