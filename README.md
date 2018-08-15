# taoist
道是万物的本源，有了道就有了一切。同样有了这样的小框架，可以组装出任意的复杂事物。
我们基于Eigen练习开发了一套训练框架。

Eigen的地址是：http://eigen.tuxfamily.org/index.php

文档地址：http://eigen.tuxfamily.org/dox/

![](http://oq8vlupt4.bkt.clouddn.com/tao.png)

# TODO List
- [x] 基础的forward,backward  @liushuchun
- [ ] dataloader模块   @lianqi
- [ ] 训练模块 
- [ ] python 调用
- [ ] 支持relu等各种激活函数 @liushuchun
- [ ] conv卷积层
- [ ] 其他一些类似LSTM等结构
- [ ] 添加nlp的一些模块，例如分词、清洗等。
- [ ] 添加图片相关的训练载入

# 运行
`cmake . && make -j4`

输出
```
graph G{
 rankdir=LR;
 nodesep=.05;
 N0 [label="x = inputs(2,1)"];
 N1 [label="y = inputs(1,1)"];
 N2 [label="a = params(1,1)"];
 N3 [label="b = params(3,1)"];
 N4 [label="W = params(3,2)"];
 N5 [label="V = params(1,3)"];
 N6 [label="W^2 = square(W)"];
 N7 [label="V^2 = square(V)"];
 N8 [label="t1 = W^2 * x"];
 N9 [label="t2 = V^2 * t1"];
 N10 [label="f = W * x"];
 N11 [label="g = f + b"];
 N12 [label="h = tan(g)"];
 N13 [label="p = V * h"];
 N14 [label="y_pred = p + a + a"];
 N15 [label="err = || y_pred - y ||^2"];
 N4 ->N6;
 N5 ->N7;
 N6 ->N8;
 N0 ->N8;
 N7 ->N9;
 N8 ->N9;
 N4 ->N10;
 N0 ->N10;
 N10 ->N11;
 N3 ->N11;
 N11 ->N12;
 N5 ->N13;
 N12 ->N13;
 N13 ->N14;
 N2 ->N14;
 N9 ->N14;
 N14 ->N15;
 N1 ->N15;
}
E = 0.585709
dE/db= -1.53063
de/da=  0.121727   -0.3352 -0.346475
df/dw=
 0.123687 -0.119318
-0.350017  0.337023
-0.231677 -0.418841
de/dv=
-0.118882 -0.110099  0.363857
```