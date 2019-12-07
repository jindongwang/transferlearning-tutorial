# 第4章 基础知识

本部分介绍迁移学习领域的一些基本知识。我们对迁移学习的问题进行简单的形式化，给出迁移学习的总体思路，并且介绍目前常用的一些度量准则。本部分中出现的所有符号和表示形式，是以后章节的基础。已有相关知识的读者可以直接跳过。



\subsection{总体思路}

形式化之后，我们可以进行迁移学习的研究。迁移学习的总体思路可以概括为：\textit{开发算法来最大限度地利用有标注的领域的知识，来辅助目标领域的知识获取和学习}。

迁移学习的核心是，找到源领域和目标领域之间的\textbf{相似性}，并加以合理利用。这种相似性非常普遍。比如，不同人的身体构造是相似的；自行车和摩托车的骑行方式是相似的；国际象棋和中国象棋是相似的；羽毛球和网球的打球方式是相似的。这种相似性也可以理解为\textbf{不变量}。以不变应万变，才能立于不败之地。

举一个杨强教授经常举的例子来说明：我们都知道在中国大陆开车时，驾驶员坐在左边，靠马路右侧行驶。这是基本的规则。然而，如果在英国、香港等地区开车，驾驶员是坐在右边，需要靠马路左侧行驶。那么，如果我们从中国大陆到了香港，应该如何快速地适应他们的开车方式呢？诀窍就是找到这里的不变量：\textit{不论在哪个地区，驾驶员都是紧靠马路中间。}这就是我们这个开车问题中的不变量。

找到相似性(不变量)，是进行迁移学习的核心。

有了这种相似性后，下一步工作就是，\textit{如何度量和利用这种相似性}。度量工作的目标有两点：一是很好地度量两个领域的相似性，不仅定性地告诉我们它们是否相似，更\textit{定量}地给出相似程度。二是以度量为准则，通过我们所要采用的学习手段，增大两个领域之间的相似性，从而完成迁移学习。

\textbf{一句话总结：} \textit{相似性是核心，度量准则是重要手段}。

\subsection{度量准则}

度量不仅是机器学习和统计学等学科中使用的基础手段，也是迁移学习中的重要工具。它的核心就是衡量两个数据域的差异。计算两个向量（点、矩阵）的距离和相似度是许多机器学习算法的基础，有时候一个好的距离度量就能决定算法最后的结果好坏。比如KNN分类算法就对距离非常敏感。本质上就是找一个变换使得源域和目标域的距离最小（相似度最大）。所以，相似度和距离度量在机器学习中非常重要。

这里给出常用的度量手段，它们都是迁移学习研究中非常常见的度量准则。对这些准则有很好的理解，可以帮助我们设计出更加好用的算法。用一个简单的式子来表示，度量就是描述源域和目标域这两个领域的距离：

\begin{equation}
	\label{eq-distance}
	DISTANCE(\mathcal{D}_s,\mathcal{D}_t) = \mathrm{DistanceMeasure}(\cdot,\cdot)
\end{equation}

下面我们从距离和相似度度量准则几个方面进行简要介绍。

\subsubsection{常见的几种距离}

\textbf{1. 欧氏距离}

定义在两个向量(空间中的两个点)上：点$$\mathbf{x}$$和点$$\mathbf{y}$$的欧氏距离为：

\begin{equation}
	\label{eq-dist-eculidean}
	d_{Euclidean}=\sqrt{(\mathbf{x}-\mathbf{y})^\top (\mathbf{x}-\mathbf{y})}
\end{equation}


\textbf{2. 闵可夫斯基距离} 

Minkowski distance， 两个向量（点）的$$p$$阶距离：

\begin{equation}
	\label{eq-dist-minkowski}
	d_{Minkowski}=(||\mathbf{x}-\mathbf{y}||^p)^{1/p}
\end{equation}

当$$p=1$$时就是曼哈顿距离，当$$p=2$$时就是欧氏距离。

\textbf{3. 马氏距离}

定义在两个向量(两个点)上，这两个数据在同一个分布里。点$$\mathbf{x}$$和点$$\mathbf{y}$$的马氏距离为：

\begin{equation}
	\label{eq-dist-maha}
	d_{Mahalanobis}=\sqrt{(\mathbf{x}-\mathbf{y})^\top \Sigma^{-1} (\mathbf{x}-\mathbf{y})}
\end{equation}

其中，$$\Sigma$$是这个分布的协方差。

当$$\Sigma=\mathbf{I}$$时，马氏距离退化为欧氏距离。

\subsubsection{相似度}

\textbf{1. 余弦相似度}

衡量两个向量的相关性(夹角的余弦)。向量$$\mathbf{x},\mathbf{y}$$的余弦相似度为：

\begin{equation}
	\label{eq-dist-cosine}
	\cos (\mathbf{x},\mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{|\mathbf{x}|\cdot |\mathbf{y}|}
\end{equation}

\textbf{2. 互信息}

定义在两个概率分布$$X,Y$$上，$$x \in X, y \in Y$$。它们的互信息为：

\begin{equation}
	\label{eq-dist-info}
	I(X;Y)=\sum_{x \in X} \sum_{y \in Y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
\end{equation}

\textbf{3. 皮尔逊相关系数}

衡量两个随机变量的相关性。随机变量$$X,Y$$的Pearson相关系数为：

\begin{equation}
	\label{eq-dist-pearson}
	\rho_{X,Y}=\frac{Cov(X,Y)}{\sigma_X \sigma_Y}
\end{equation}

理解：协方差矩阵除以标准差之积。

范围：$$[-1,1]$$，绝对值越大表示（正/负）相关性越大。

\textbf{4. Jaccard相关系数}

对两个集合$$X,Y$$，判断他们的相关性，借用集合的手段：

\begin{equation}
	\label{eq-dist-jaccard}
	J=\frac{X \cap Y}{X \cup Y}
\end{equation}

理解：两个集合的交集除以并集。

扩展：Jaccard距离=$$1-J$$。

\subsubsection{KL散度与JS距离}

KL散度和JS距离是迁移学习中被广泛应用的度量手段。

\textbf{1. KL散度}

Kullback–Leibler divergence，又叫做\textit{相对熵}，衡量两个概率分布$$P(x),Q(x)$$的距离：

\begin{equation}
	\label{eq-dist-kl}
	D_{KL}(P||Q)=\sum_{i=1} P(x) \log \frac{P(x)}{Q(x)}
\end{equation}

这是一个非对称距离：$$D_{KL}(P||Q) \ne D_{KL}(Q||P)$$.

\textbf{2. JS距离}

Jensen–Shannon divergence，基于KL散度发展而来，是对称度量：

\begin{equation}
	\label{eq-dist-js}
	JSD(P||Q)= \frac{1}{2} D_{KL}(P||M) + \frac{1}{2} D_{KL}(Q||M)
\end{equation}

其中$$M=\frac{1}{2}(P+Q)$$。

\subsubsection{最大均值差异MMD}

最大均值差异是迁移学习中使用频率最高的度量。Maximum mean discrepancy，它度量在再生希尔伯特空间中两个分布的距离，是一种核学习方法。两个随机变量的MMD平方距离为

\begin{equation}
	\label{eq-dist-mmd}
	MMD^2(X,Y)=\left \Vert \sum_{i=1}^{n_1}\phi(\mathbf{x}_i)- \sum_{j=1}^{n_2}\phi(\mathbf{y}_j) \right \Vert^2_\mathcal{H}
\end{equation}

其中$$\phi(\cdot)$$是映射，用于把原变量映射到\textit{再生核希尔伯特空间}(Reproducing Kernel Hilbert Space, RKHS)~\cite{borgwardt2006integrating}中。什么是RKHS？形式化定义太复杂，简单来说希尔伯特空间是对于函数的内积完备的，而再生核希尔伯特空间是具有再生性$$\langle K(x,\cdot),K(y,\cdot)\rangle_\mathcal{H}=K(x,y)$$的希尔伯特空间。就是比欧几里得空间更高端的。将平方展开后，RKHS空间中的内积就可以转换成核函数，所以最终MMD可以直接通过核函数进行计算。

理解：就是求两堆数据在RKHS中的\textit{均值}的距离。

\textit{Multiple-kernel MMD}：多核的MMD，简称MK-MMD。现有的MMD方法是基于单一核变换的，多核的MMD假设最优的核可以由多个核线性组合得到。多核MMD的提出和计算方法在文献~\cite{gretton2012optimal}中形式化给出。MK-MMD在许多后来的方法中被大量使用，最著名的方法是DAN~\cite{long2015learning}。我们将在后续单独介绍此工作。

\subsubsection{Principal Angle}

也是将两个分布映射到高维空间(格拉斯曼流形)中，在流形中两堆数据就可以看成两个点。Principal angle是求这两堆数据的对应维度的夹角之和。

对于两个矩阵$$\mathbf{X},\mathbf{Y}$$，计算方法：首先正交化(用PCA)两个矩阵，然后：

\begin{equation}
\label{eq-dist-pa}
PA(\mathbf{X},\mathbf{Y})=\sum_{i=1}^{\min(m,n)} \sin \theta_i
\end{equation}

其中$$m,n$$分别是两个矩阵的维度，$$\theta_i$$是两个矩阵第$$i$$个维度的夹角，$$\Theta=\{\theta_1,\theta_2,\cdots,\theta_t\}$$是两个矩阵SVD后的角度：

\begin{equation}
	\mathbf{X}^\top\mathbf{Y}=\mathbf{U} (\cos \Theta) \mathbf{V}^\top
\end{equation}

\subsubsection{A-distance}

$$\mathcal{A}$$-distance是一个很简单却很有用的度量。文献\cite{ben2007analysis}介绍了此距离，它可以用来估计不同分布之间的差异性。$$\mathcal{A}$$-distance被定义为建立一个线性分类器来区分两个数据领域的hinge损失(也就是进行二类分类的hinge损失)。它的计算方式是，我们首先在源域和目标域上训练一个二分类器$$h$$，使得这个分类器可以区分样本是来自于哪一个领域。我们用$$err(h)$$来表示分类器的损失，则$$\mathcal{A}$$-distance定义为：

\begin{equation}
	\label{eq-dist-adist}
	\mathcal{A}(\mathcal{D}_s,\mathcal{D}_t) = 2(1 - 2 err(h))
\end{equation}

$$\mathcal{A}$$-distance通常被用来计算两个领域数据的相似性程度，以便与实验结果进行验证对比。

\subsubsection{Hilbert-Schmidt Independence Criterion}

希尔伯特-施密特独立性系数，Hilbert-Schmidt Independence Criterion，用来检验两组数据的独立性：
\begin{equation}
	HSIC(X,Y) = trace(HXHY)
\end{equation}
其中$$X,Y$$是两堆数据的kernel形式。

\subsubsection{Wasserstein Distance}

Wasserstein Distance是一套用来衡量两个概率分部之间距离的度量方法。该距离在一个度量空间$$(M,\rho)$$上定义，其中$$\rho(x,y)$$表示集合$$M$$中两个实例$$x$$和$$y$$的距离函数，比如欧几里得距离。两个概率分布$$\mathbb{P}$$和$$\mathbb{Q}$$之间的$$p{\text{-th}}$$ Wasserstein distance可以被定义为

\begin{equation}
W_p(\mathbb{P}, \mathbb{Q}) = \Big(\inf_{\mu \in \Gamma(\mathbb{P}, \mathbb{Q}) } \int \rho(x,y)^p d\mu(x,y) \Big)^{1/p},
\end{equation}

其中$$\Gamma(\mathbb{P}, \mathbb{Q})$$是在集合$$M\times M$$内所有的以$$\mathbb{P}$$和$$\mathbb{Q}$$为边缘分布的联合分布。著名的Kantorovich-Rubinstein定理表示当$$M$$是可分离的时候，第一Wasserstein distance可以等价地表示成一个积分概率度量(integral probability metric)的形式

\begin{equation}
W_1(\mathbb{P},\mathbb{Q})= \sup_{\left \| f \right \|_L \leq 1} \mathbb{E}_{x \sim \mathbb{P}}[f(x)] - \mathbb{E}_{x \sim \mathbb{Q}}[f(x)],
\end{equation}
其中$$\left \| f \right \|_L = \sup{|f(x) - f(y)|} / \rho(x,y)$$并且$$\left \| f \right \|_L \leq 1$$称为$$1-$$利普希茨条件。

\subsection{迁移学习的理论保证*}
\textit{
本部分的标题中带有*号，有一些难度，为可看可不看的内容。此部分最常见的形式是当自己提出的算法需要理论证明时，可以借鉴。}

在第一章里我们介绍了两个重要的概念：迁移学习是什么，以及为什么需要迁移学习。但是，还有一个重要的问题没有得到解答：\textit{为什么可以进行迁移}?也就是说，迁移学习的可行性还没有探讨。

值得注意的是，就目前的研究成果来说，迁移学习领域的理论工作非常匮乏。我们在这里仅回答一个问题：为什么数据分布不同的两个领域之间，知识可以进行迁移？或者说，到底达到什么样的误差范围，我们才认为知识可以进行迁移？

加拿大滑铁卢大学的Ben-David等人从2007年开始，连续发表了三篇文章~\cite{ben2007analysis,blitzer2008learning,ben2010theory}对迁移学习的理论进行探讨。在文中，作者将此称之为“Learning from different domains”。在三篇文章也成为了迁移学习理论方面的经典文章。文章主要回答的问题就是：在怎样的误差范围内，从不同领域进行学习是可行的？

\textbf{学习误差：} 给定两个领域$$\mathcal{D}_s,\mathcal{D}_t$$，$$X$$是定义在它们之上的数据，一个假设类$$\mathcal{H}$$。则两个领域$$\mathcal{D}_s,\mathcal{D}_t$$之间的$$\mathcal{H}$$-divergence被定义为

\begin{equation}
	\hat{d}_{\mathcal{H}}(\mathcal{D}_s,\mathcal{D}_t) = 2 \sup_{\eta \in \mathcal{H}} \left|\underset{\mathbf{x} \in \mathcal{D}_s}{P}[\eta(\mathbf{x}) = 1] - \underset{\mathbf{x} \in \mathcal{D}_t}{P}[\eta(\mathbf{x}) = 1] \right|
\end{equation}

因此，这个$$\mathcal{H}$$-divergence依赖于假设$$\mathcal{H}$$来判别数据是来自于$$\mathcal{D}_s$$还是$$\mathcal{D}_t$$。作者证明了，对于一个对称的$$\mathcal{H}$$，我们可以通过如下的方式进行计算

\begin{equation}
	d_\mathcal{H} (\mathcal{D}_s,\mathcal{D}_t) = 2 \left(1 - \min_{\eta \in \mathcal{H}} \left[\frac{1}{n_1} \sum_{i=1}^{n_1} I[\eta(\mathbf{x}_i)=0] + \frac{1}{n_2} \sum_{i=1}^{n_2} I[\eta(\mathbf{x}_i)=1]\right] \right)
\end{equation}
其中$$I[a]$$为指示函数：当$$a$$成立时其值为1,否则其值为0。

\textbf{在目标领域的泛化界：}

假设$$\mathcal{H}$$为一个具有$$d$$个VC维的假设类，则对于任意的$$\eta \in \mathcal{H}$$，下面的不等式有$$1 - \delta$$的概率成立：

\begin{equation}
	R_{\mathcal{D}_t}(\eta) \le R_s(\eta) + \sqrt{\frac{4}{n}(d \log \frac{2en}{d} + \log \frac{4}{\delta})} + \hat{d}_{\mathcal{H}}(\mathcal{D}_s,\mathcal{D}_t) + 4 \sqrt{\frac{4}{n}(d \log \frac{2n}{d} + \log \frac{4}{\delta})} + \beta
\end{equation}
其中
\begin{equation}
	\beta \ge \inf_{\eta^\star \in \mathcal{H}} [R_{\mathcal{D}_s}(\eta^\star) + R_{\mathcal{D}_t}(\eta^\star)]
\end{equation}
并且
\begin{equation}
	R_{s}(\eta) = \frac{1}{n} \sum_{i=1}^{m} I[\eta(\mathbf{x}_i) \ne y_i]
\end{equation}

具体的理论证明细节，请参照上述提到的三篇文章。

在自己的研究中，如果需要进行相关的证明，可以参考一些已经发表的文章的写法，例如~\cite{long2014adaptation}等。

另外，英国的Gretton等人也在进行一些学习理论方面的研究，有兴趣的读者可以关注他的个人主页：\url{http://www.gatsby.ucl.ac.uk/~gretton/}。
