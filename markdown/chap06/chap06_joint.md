# 联合分布自适应

## 基本思路

联合分布自适应方法(Joint Distribution Adaptation)的目标是减小源域和目标域的联合概率分布的距离，从而完成迁移学习。从形式上来说，联合分布自适应方法是用$$P(\mathbf{x}_s)$$和$$P(\mathbf{x}_t)$$之间的距离、以及$$P(y_s|\mathbf{x}_s)$$和$$P(y_t|\mathbf{x}_t)$$之间的距离来近似两个领域之间的差异。即：

$$
DISTANCE(\mathcal{D}_s,\mathcal{D}_t) \approx ||P(\mathbf{x}_s) - P(\mathbf{x}_t)|| + ||P(y_s|\mathbf{x}_s) - P(y_t|\mathbf{x}_t)||
$$

## 核心方法

联合分布适配的[JDA方法](http://openaccess.thecvf.com/content_iccv_2013/html/Long_Transfer_Feature_Learning_2013_ICCV_paper.html)首次发表于2013年的ICCV(计算机视觉领域顶会，与CVPR类似)。

假设是最基本的出发点。那么JDA这个方法的假设是什么呢？就是假设两点：1）源域和目标域边缘分布不同，2）源域和目标域条件分布不同。既然有了目标，同时适配两个分布不就可以了吗？于是作者很自然地提出了联合分布适配方法：适配联合概率。

不过这里我感觉有一些争议：边缘分布和条件分布不同，与联合分布不同并不等价。所以这里的“联合”二字实在是会引起歧义。我的理解是，同时适配两个分布，也可以叫联合，而不是概率上的“联合”。尽管作者在文章里第一个公式就写的是适配联合概率，但是这里感觉是有一些问题的。我们抛开它这个有歧义的，把“联合”理解成同时适配两个分布。

那么，JDA方法的目标就是，寻找一个变换$$\mathbf{A}$$，使得经过变换后的$$P(\mathbf{A}^\top \mathbf{x}_s)$$和$$P(\mathbf{A}^\top \mathbf{x}_t)$$的距离能够尽可能地接近，同时，$$P(y_s|\mathbf{A}^\top \mathbf{x}_s)$$和$$P(y_t|\mathbf{A}^\top \mathbf{x}_t)$$的距离也要小。很自然地，这个方法也就分成了两个步骤。

### 边缘分布适配

首先来适配边缘分布，也就是$$P(\mathbf{A}^\top \mathbf{x}_s)$$和$$P(\mathbf{A}^\top \mathbf{x}_t)$$的距离能够尽可能地接近。其实这个操作就是迁移成分分析(TCA)。我们仍然使用MMD距离来最小化源域和目标域的最大均值差异。MMD距离是

$$
	\left \Vert \frac{1}{n} \sum_{i=1}^{n} \mathbf{A}^\top \mathbf{x}_{i} - \frac{1}{m} \sum_{j=1}^{m} \mathbf{A}^\top \mathbf{x}_{j} \right \Vert ^2_\mathcal{H}
$$

这个式子实在不好求解。我们引入核方法，化简这个式子，它就变成了

$$
	D(\mathcal{D}_s,\mathcal{D}_t)=tr(\mathbf{A}^\top \mathbf{X} \mathbf{M}_0 \mathbf{X}^\top \mathbf{A})
$$

其中$$\mathbf{A}$$就是变换矩阵，我们把它加黑加粗，$$\mathbf{X}$$是源域和目标域合并起来的数据。$$\mathbf{M}_0$$是一个MMD矩阵：

$$
	(\mathbf{M}_0)_{ij}=\begin{cases} \frac{1}{n^2}, & \mathbf{x}_i,\mathbf{x}_j \in \mathcal{D}_s\\ \frac{1}{m^2}, & \mathbf{x}_i,\mathbf{x}_j \in \mathcal{D}_t\\ -\frac{1}{mn}, & \text{otherwise} \end{cases}
$$

$$n,m$$分别是源域和目标域样本的个数。

到此为止没有什么创新点，因为这就是一个TCA。

### 条件分布适配

这是我们要做的第二个目标，适配源域和目标域的条件概率分布。也就是说，还是要找一个变换$$\mathbf{A}$$，使得$$P(y_s|\mathbf{A}^\top \mathbf{x}_s)$$和$$P(y_t|\mathbf{A}^\top \mathbf{x}_t)$$的距离也要小。那么简单了，我们再用一遍MMD啊。可是问题来了：我们的目标域里，没有$$y_t$$，没法求目标域的条件分布！

这条路看来是走不通了。也就是说，直接建模$$P(y_t|\mathbf{x}_t)$$不行。那么，能不能有别的办法可以逼近这个条件概率？我们可以换个角度，利用类条件概率$$P(\mathbf{x}_t|y_t)$$。根据贝叶斯公式$$P(y_t|\mathbf{x}_t)=p(y_t)p(\mathbf{x}_t|y_t)$$，我们如果忽略$$P(\mathbf{x}_t)$$，那么岂不是就可以用$$P(\mathbf{x}_t|y_t)$$来近似$$P(y_t|\mathbf{x}_t)$$？

而这样的近似也不是空穴来风。在统计学上，有一个概念叫做**充分统计量**，它是什么意思呢？大概意思就是说，如果样本里有太多的东西未知，样本足够好，我们就能够从中选择一些统计量，近似地代替我们要估计的分布。好了，我们为近似找到了理论依据。

实际怎么做呢？我们依然没有$$y_t$$。采用的方法是，用$$(\mathbf{x}_s,y_s)$$来训练一个简单的分类器(比如knn、逻辑斯特回归)，到$$\mathbf{x}_t$$上直接进行预测。总能够得到一些伪标签$$\hat{y}_t$$。我们根据伪标签来计算，这个问题就可解了。

类与类之间的MMD距离表示为

$$
	\sum_{c=1}^{C}\left \Vert \frac{1}{n_c} \sum_{\mathbf{x}_{i} \in \mathcal{D}^{(c)}_s} \mathbf{A}^\top \mathbf{x}_{i} - \frac{1}{m_c} \sum_{\mathbf{x}_{i} \in \mathcal{D}^{(c)}_t} \mathbf{A}^\top \mathbf{x}_{i} \right \Vert ^2_\mathcal{H}
$$

其中，$$n_c,m_c$$分别标识源域和目标域中来自第$$c$$类的样本个数。同样地我们用核方法，得到了下面的式子

$$
	\sum_{c=1}^{C}tr(\mathbf{A}^\top \mathbf{X} \mathbf{M}_c \mathbf{X}^\top \mathbf{A})
$$

其中$$\mathbf{M}_c$$为

$$
	(\mathbf{M}_c)_{ij}=\begin{cases} \frac{1}{n^2_c}, & \mathbf{x}_i,\mathbf{x}_j \in \mathcal{D}^{(c)}_s\\ \frac{1}{m^2_c}, & \mathbf{x}_i,\mathbf{x}_j \in \mathcal{D}^{(c)}_t\\ -\frac{1}{m_c n_c}, & \begin{cases} \mathbf{x}_i \in \mathcal{D}^{(c)}_s ,\mathbf{x}_j \in \mathcal{D}^{(c)}_t \\ \mathbf{x}_i \in \mathcal{D}^{(c)}_t ,\mathbf{x}_j \in \mathcal{D}^{(c)}_s \end{cases}\\ 0, & \text{otherwise}\end{cases}
$$

### 总优化目标

现在我们把两个距离结合起来，得到了一个总的优化目标：

$$
	\min \sum_{c=0}^{C}tr(\mathbf{A}^\top \mathbf{X} \mathbf{M}_c \mathbf{X}^\top \mathbf{A}) + \lambda \Vert \mathbf{A} \Vert ^2_F
$$

看到没，通过$$c=0 \cdots C$$就把两个距离统一起来了！其中的$$\lambda \Vert \mathbf{A} \Vert ^2_F$$是正则项，使得模型是**良好定义(Well-defined)**的。

我们还缺一个限制条件，不然这个问题无法解。限制条件是什么呢？和TCA一样，变换前后数据的方差要维持不变。怎么求数据的方差呢，还和TCA一样：$$\mathbf{A}^\top \mathbf{X} \mathbf{H} \mathbf{X}^\top \mathbf{A} = \mathbf{I}$$，其中的$$\mathbf{H}$$也是中心矩阵，$$\mathbf{I}$$是单位矩阵。也就是说，我们又添加了一个优化目标是要$$\max \mathbf{A}^\top \mathbf{X} \mathbf{H} \mathbf{X}^\top \mathbf{A}$$(这一个步骤等价于PCA了)。和原来的优化目标合并，优化目标统一为：

$$
	\min \frac{\sum_{c=0}^{C}tr(\mathbf{A}^\top \mathbf{X} \mathbf{M}_c \mathbf{X}^\top \mathbf{A}) + \lambda \Vert \mathbf{A}\Vert^2_F}{ \mathbf{A}^\top \mathbf{X} \mathbf{H} \mathbf{X}^\top \mathbf{A}}
$$

这个式子实在不好求解。但是，有个东西叫做[Rayleigh quotient](https://www.wikiwand.com/en/Rayleigh_quotient)，上面两个一样的这种形式。因为$$\mathbf{A}$$是可以进行拉伸而不改变最终结果的，而如果下面为0的话，整个式子就求不出来值了。所以，我们直接就可以让下面不变，只求上面。所以我们最终的优化问题形式搞成了

$$
	\min \quad \sum_{c=0}^{C}tr(\mathbf{A}^\top \mathbf{X} \mathbf{M}_c \mathbf{X}^\top \mathbf{A}) + \lambda \Vert \mathbf{A} \Vert ^2_F \quad \text{s.t.} \quad \mathbf{A}^\top \mathbf{X} \mathbf{H} \mathbf{X}^\top \mathbf{A} = \mathbf{I}
$$

怎么解？太简单了，可以用拉格朗日法。最后变成了

$$
	\left(\mathbf{X} \sum_{c=0}^{C} \mathbf{M}_c \mathbf{X}^\top + \lambda \mathbf{I}\right) \mathbf{A} =\mathbf{X} \mathbf{H} \mathbf{X}^\top \mathbf{A} \Phi 
$$

其中的$$\Phi$$是拉格朗日乘子。别看这个东西复杂，又有要求解的$$\mathbf{A}$$，又有一个新加入的$$\Phi$$ 。但是它在Matlab里是可以直接解的(用$$\mathrm{eigs}$$函数即可)。这样我们就得到了变换$$\mathbf{A}$$，问题解决了。

可是伪标签终究是伪标签啊，肯定精度不高，怎么办？有个东西叫做\textit{迭代}，一次不行，我们再做一次。后一次做的时候，我们用上一轮得到的标签来作伪标签。这样的目的是得到越来越好的伪标签，而参与迁移的数据是不会变的。这样往返多次，结果就自然而然好了。

## 扩展

JDA方法是十分经典的迁移学习方法。后续的相关工作通过在JDA的基础上加入额外的损失项，使得迁移学习的效果得到了很大提升。我们在这里简要介绍一些基于JDA的相关工作。

- [ARTL (Adaptation Regularization)](https://ieeexplore.ieee.org/abstract/document/6550016/): 将JDA嵌入一个结构风险最小化框架中，用表示定理直接学习分类器
- [VDA](https://link.springer.com/article/10.1007/s10115-016-0944-x): 在JDA的优化目标中加入了类内距和类间距的计算
- [hsiao2016learning](https://ieeexplore.ieee.org/abstract/document/7478127/): 在JDA的基础上加入结构不变性控制
- [hou2015unsupervised](https://ieeexplore.ieee.org/abstract/document/7301758/)： 在JDA的基础上加入目标域的选择
- [JGSA (Joint Geometrical and Statistical Alignment)](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Joint_Geometrical_and_CVPR_2017_paper.html): 在JDA的基础上加入类内距、类间距、标签持久化
- [JAN~(Joint Adaptation Network)(https://dl.acm.org/citation.cfm?id=3305909)]: 提出了联合分布度量JMMD，在深度网络中进行联合分布的优化

JDA的代码可以在这里被找到：https://github.com/jindongwang/transferlearning/tree/master/code/traditional/JDA。