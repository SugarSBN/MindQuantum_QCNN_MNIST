## 关于实现的一点说明

* 山东大学 2020级 苏博南 
* www.subonan.com 

----

### 文件说明

* tools.py 这里面主要有两个函数：
  * >resize(a, lenb)

  这其实是我找同学写的一个小算法hhh。给出一个$28\times 28$的方阵a，返回一个$lenb\times lenb$的方阵。因为懒得装openCV，于是手动写了一个把图像降低分辨率的操作，原理是根据几何关系计算新的像素值。总之可以把$28\times28$的MNIST原始数据改成$16\times 16$的，这就可以用8个比特encode了。写的可能比较丑陋，但毕竟只是初始化数据用的，所以能用一次性就行了。
  * >controlled_gate(circuit, gate, tqubit, cqubits, zero_qubit)

  这其实是因为我发现mindquantum的受控门不支持“空心受控”，于是我就自己写了个。如果$cqubits=[0,-1,2],zero\_qubit=1$就表示第一、三量子比特为1时，第二量子比特为0时，才进行运算。就是用负数表示了“空心受控”，然后zero_qubit单独判断了下第0量子比特。（因为它没有符号）

* test.py: 主程序，里面写注释了。用来训练和预测。

* MNIST_params.npy：因为我手动写了个实现amplitude encoding的线路，然后该线路也是有参数的，参数是原始数据的各种复合运算。因为我不知道mindquantum支持参数复合运算的操作，于是我就预处理了下数据，把这些参数都算出来了（针对每个样本）。原始数据是$16\times 16=[0,256)$，但encoder中的参数只有$[0,255)$。所以MNIST_params.npy里的数据的规模是$60000\times 255$的。
* MNIST_train.npy：$60000\times 256$的原始数据
* MNIST_train_formalized：$60000\times 256$的原始数据，每行都归一化了，方便encoding。
* weights.npy：我已经训练好的一组ansatz的参数值，在test.py中有用它来预测。

### 关于原理的一些说明

* encoder部分是我自己想的hhh，我想了一个可以实现amplitude encoding的线路。复杂度不高。

* ansatz我选的是https://arxiv.org/pdf/2108.00661.pdf中讲到的convolutional circuit1：

  ![cc](./cc.png)

  pooling层就比较随意的受控RZ和受控RX门

* 最后就是基于计算基下测量。

* 然后就是Adam优化啥的都是自带的好东西，一直train就完事了