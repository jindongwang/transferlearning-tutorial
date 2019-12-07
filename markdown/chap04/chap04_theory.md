# 迁移学习的理论保证*


本部分的标题中带有*号，有一些难度，为可看可不看的内容。此部分最常见的形式是当自己提出的算法需要理论证明时，可以借鉴。

在第一章里我们介绍了两个重要的概念：迁移学习是什么，以及为什么需要迁移学习。但是，还有一个重要的问题没有得到解答：*为什么可以进行迁移*?也就是说，迁移学习的可行性还没有探讨。

值得注意的是，就目前的研究成果来说，迁移学习领域的理论工作非常匮乏。我们在这里仅回答一个问题：为什么数据分布不同的两个领域之间，知识可以进行迁移？或者说，到底达到什么样的误差范围，我们才认为知识可以进行迁移？

加拿大滑铁卢大学的Ben-David等人从2007年开始，连续地对迁移学习的理论进行探讨。具体的一些文章可以见[这里](https://github.com/jindongwang/transferlearning#3theory-and-survey-%E7%90%86%E8%AE%BA%E4%B8%8E%E7%BB%BC%E8%BF%B0)在文中，作者将此称之为“Learning from different domains”。在三篇文章也成为了迁移学习理论方面的经典文章。文章主要回答的问题就是：在怎样的误差范围内，从不同领域进行学习是可行的？

**学习误差：** 给定两个领域$$\mathcal{D}_s,\mathcal{D}_t$$，$$X$$是定义在它们之上的数据，一个假设类$$\mathcal{H}$$。则两个领域$$\mathcal{D}_s,\mathcal{D}_t$$之间的$$\mathcal{H}$$-divergence被定义为

$$
	\hat{d}_{\mathcal{H}}(\mathcal{D}_s,\mathcal{D}_t) = 2 \sup_{\eta \in \mathcal{H}} \left|\underset{\mathbf{x} \in \mathcal{D}_s}{P}[\eta(\mathbf{x}) = 1] - \underset{\mathbf{x} \in \mathcal{D}_t}{P}[\eta(\mathbf{x}) = 1] \right|
$$

因此，这个$$\mathcal{H}$$-divergence依赖于假设$$\mathcal{H}$$来判别数据是来自于$$\mathcal{D}_s$$还是$$\mathcal{D}_t$$。作者证明了，对于一个对称的$$\mathcal{H}$$，我们可以通过如下的方式进行计算

$$
	d_\mathcal{H} (\mathcal{D}_s,\mathcal{D}_t) = 2 \left(1 - \min_{\eta \in \mathcal{H}} \left[\frac{1}{n_1} \sum_{i=1}^{n_1} I[\eta(\mathbf{x}_i)=0] + \frac{1}{n_2} \sum_{i=1}^{n_2} I[\eta(\mathbf{x}_i)=1]\right] \right)
$$
其中$$I[a]$$为指示函数：当$$a$$成立时其值为1,否则其值为0。

**在目标领域的泛化界(Bound)：**

假设$$\mathcal{H}$$为一个具有$$d$$个VC维的假设类，则对于任意的$$\eta \in \mathcal{H}$$，下面的不等式有$$1 - \delta$$的概率成立：

$$
	R_{\mathcal{D}_t}(\eta) \le R_s(\eta) + \sqrt{\frac{4}{n}(d \log \frac{2en}{d} + \log \frac{4}{\delta})} + \hat{d}_{\mathcal{H}}(\mathcal{D}_s,\mathcal{D}_t) + 4 \sqrt{\frac{4}{n}(d \log \frac{2n}{d} + \log \frac{4}{\delta})} + \beta
$$
其中
$$
	\beta \ge \inf_{\eta^\star \in \mathcal{H}} [R_{\mathcal{D}_s}(\eta^\star) + R_{\mathcal{D}_t}(\eta^\star)]
$$
并且
$$
	R_{s}(\eta) = \frac{1}{n} \sum_{i=1}^{m} I[\eta(\mathbf{x}_i) \ne y_i]
$$

具体的理论证明细节，请参照上述提到的三篇文章。

在自己的研究中，如果需要进行相关的证明，可以参考一些已经发表的文章的写法，例如[Adaptation regularization: a general framework for transfer learning](https://ieeexplore.ieee.org/abstract/document/6550016/)等。

另外，英国的Gretton等人也在进行一些学习理论方面的研究，有兴趣的读者可以关注他的个人主页：http://www.gatsby.ucl.ac.uk/~gretton/。