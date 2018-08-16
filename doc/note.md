#softmax
softmax：重新定义了多层神经网络的输出层（output layer），注意仅和输出层有关系，和其他层无关。

softmax function，也称为 normalized exponential（指数族分布的观点）；
1. softmax
我们知道在神经网络的前馈（feedforward）的过程中，输出层的输入（input）为： 
在 softmax 的机制中，为获得输出层的输出（也即最终的输出），我们不是将 sigmoid 函数作用于其上， 
而是采用所谓的 softmax function：

因此：

（1）输出层输出之和为 1 
因为输出层的输出之和为1，其中一项增加，其他所有项则会相应减少。

（2）输出层全部输出均为正： 
而且 softmax 的机制，也保证了所有的输出均为正值；

终上所述，softmax 层的输出其实为一种概率分布（probability distribution），因此对于一个多 label 的分类任务（比如手写字符识别，0-9）而言， 对应于最终的分类结果为  的概率。

2. logsoftmax
将原始数据从 x ⇒ log (x)，无疑会原始数据的值域进行一定的收缩。

进一步地，还可对原始数据进行进一步的预处理，

# 假设 x 是一个向量
```python
def logsoftmax(x):
    m = T.max(x)
    exp_x = T.exp(x-m)
    Z = T.sum(exp_x)
    return x-m-T.log(Z)
```


